import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from deeprpi.model.embedding import load_esm, load_rnabert, ESMEmbedding, RNABertEmbedding
from deeprpi.model.classifier import SimpleProteinRNAClassifier
from deeprpi.utils import RPIDataset

class CrossAttention(nn.Module):
    def __init__(self, protein_dim: int, rna_dim: int, num_heads: int = 8, dropout: float = 0.1, device: str = None):
        super().__init__()
        self.num_heads = num_heads
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # No need to adjust dimensions, use original dimensions directly
        self.protein_dim = protein_dim
        self.rna_dim = rna_dim
        
        # Protein to RNA attention
        self.protein_to_rna = nn.MultiheadAttention(
            embed_dim=self.protein_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=self.rna_dim,
            vdim=self.rna_dim
        ).to(self.device)
        
        # RNA to protein attention
        self.rna_to_protein = nn.MultiheadAttention(
            embed_dim=self.rna_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=self.protein_dim,
            vdim=self.protein_dim
        ).to(self.device)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.protein_dim).to(self.device)
        self.norm2 = nn.LayerNorm(self.rna_dim).to(self.device)
        
        # Dropout
        self.dropout = nn.Dropout(dropout).to(self.device)
        
        # Remove unnecessary projection layers
        
    def forward(self, protein_embeddings: torch.Tensor, rna_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Ensure all tensors are on the same device
        protein_embeddings = protein_embeddings.to(self.device)
        rna_embeddings = rna_embeddings.to(self.device)
        
        # Remove unnecessary shape checks and unsqueeze operations
        
        # Calculate protein to RNA attention
        protein_attended, protein_attention_weights = self.protein_to_rna(
            query=protein_embeddings,
            key=rna_embeddings,
            value=rna_embeddings,
            need_weights=True
        )
        
        # Calculate RNA to protein attention
        rna_attended, rna_attention_weights = self.rna_to_protein(
            query=rna_embeddings,
            key=protein_embeddings,
            value=protein_embeddings,
            need_weights=True
        )
        
        # Residual connection and layer normalization
        protein_embeddings = self.norm1(protein_embeddings + self.dropout(protein_attended))
        rna_embeddings = self.norm2(rna_embeddings + self.dropout(rna_attended))
        
        return protein_embeddings, rna_embeddings, protein_attention_weights, rna_attention_weights

class ProteinRNALightningModule(pl.LightningModule):
    def __init__(self, output_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.esm_embedding = ESMEmbedding(*load_esm(output_dim=output_dim), device=self.device)
        self.rna_embedding = RNABertEmbedding(*load_rnabert(output_dim=output_dim), device=self.device)
        
        # ESM-2 output dimension is 1280, RNA-BERT output dimension is 120
        protein_dim = 1280
        rna_dim = 120
        
        # Initialize classifier with correct dimensions
        self.classifier = SimpleProteinRNAClassifier(
            protein_dim=protein_dim,  # Protein embedding dimension
            rna_dim=rna_dim,         # RNA embedding dimension
            hidden_dim=hidden_dim,   # Hidden layer dimension
            dropout=dropout          # Dropout rate
        )
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Create attention map save directory in DeepRPI folder
        self.attention_dir = Path("attention_maps")
        self.attention_dir.mkdir(exist_ok=True, parents=True)
        
    def forward(self, protein_seqs, rna_seqs):
        # Get embeddings - default no pooling, preserve sequence dimension
        protein_embeddings, _, _ = self.esm_embedding(protein_seqs)
        rna_embeddings, _, _ = self.rna_embedding(rna_seqs)
        
        # Forward pass through classifier
        # The classifier will handle cross-attention, pooling, concatenation and classification internally
        logits, protein_attention, rna_attention = self.classifier(protein_embeddings, rna_embeddings)
        return logits, protein_attention, rna_attention
    
    def _process_batch(self, batch):
        rna_seqs = batch[0] 
        protein_seqs = batch[2]  
        labels = batch[4]  
        
        if len(labels.shape) > 1:
            labels = labels[:, 0]
            
        return rna_seqs, protein_seqs, labels
    
    def _plot_attention(self, attention_weights, title, save_path):
        """Plot attention heatmap and save"""
        # Check if attention weights are available
        if attention_weights is None:
            print(f"Warning: No attention weights available for {title}")
            return
        
        # Get attention matrix shape
        attention_cpu = attention_weights.cpu().detach().numpy()
        shape = attention_cpu.shape
        print(f"Attention matrix shape: {shape}")
        
        # If it's a 1×1 matrix, generate a more meaningful visualization
        if len(shape) == 2 and shape[0] == 1 and shape[1] == 1:
            print(f"Attention is a single value: {attention_cpu[0][0]}")
            plt.figure(figsize=(6, 4))
            plt.bar(['Attention'], [attention_cpu[0][0]])
            plt.title(title)
            plt.ylabel('Attention Value')
            plt.ylim(0, 1)
            plt.savefig(save_path)
            plt.close()
            return
            
        # For normal attention matrices, draw heatmap
        plt.figure(figsize=(10, 8))
        
        # Generate more meaningful axis labels
        if len(shape) == 2:
            rows, cols = shape
            # Create tick marks, select subset if dimensions are large
            if rows > 10:
                row_ticks = np.linspace(0, rows-1, min(10, rows)).astype(int)
                row_labels = [str(i) for i in row_ticks]
            else:
                row_ticks = np.arange(rows)
                row_labels = [str(i) for i in range(rows)]
                
            if cols > 10:
                col_ticks = np.linspace(0, cols-1, min(10, cols)).astype(int)
                col_labels = [str(i) for i in col_ticks]
            else:
                col_ticks = np.arange(cols)
                col_labels = [str(i) for i in range(cols)]
            
            # Draw heatmap and set ticks
            heatmap = sns.heatmap(
                attention_cpu, 
                cmap='viridis',
                annot=rows*cols <= 100,  # Show values when not too many cells
                fmt='.2f', 
                cbar=True
            )
            
            plt.xticks(col_ticks, col_labels)
            plt.yticks(row_ticks, row_labels)
            plt.xlabel('Target position')
            plt.ylabel('Source position')
        else:
            # For non-2D cases, simply show the first plane
            heatmap = sns.heatmap(
                attention_cpu.reshape(shape[0], -1), 
                cmap='viridis',
                cbar=True
            )
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)  # Increase DPI for better output quality
        plt.close()
    
    def training_step(self, batch, batch_idx):
        rna_seqs, protein_seqs, labels = self._process_batch(batch)
        logits, protein_attention, rna_attention = self(protein_seqs, rna_seqs)
        loss = self.criterion(logits.squeeze(), labels.float())
        
        # Calculate metrics
        preds = (torch.sigmoid(logits) > 0.5).int()
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        
        # Save attention maps (every 100 batches)
        if batch_idx % 100 == 0 and protein_attention is not None:
            self._plot_attention(
                protein_attention[0],  # Take attention of the first sample
                f"Protein to RNA Attention (Batch {batch_idx})",
                self.attention_dir / f"protein_attention_batch_{batch_idx}.png"
            )
            self._plot_attention(
                rna_attention[0],  # Take attention of the first sample
                f"RNA to Protein Attention (Batch {batch_idx})",
                self.attention_dir / f"rna_attention_batch_{batch_idx}.png"
            )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        rna_seqs, protein_seqs, labels = self._process_batch(batch)
        logits, protein_attention, rna_attention = self(protein_seqs, rna_seqs)
        loss = self.criterion(logits.squeeze(), labels.float())
        
        # Calculate metrics
        preds = (torch.sigmoid(logits) > 0.5).int()
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())
        
        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc, 'val_f1': f1}
    
    def test_step(self, batch, batch_idx):
        rna_seqs, protein_seqs, labels = self._process_batch(batch)
        logits, protein_attention, rna_attention = self(protein_seqs, rna_seqs)
        
        # Calculate metrics
        preds = (torch.sigmoid(logits) > 0.5).int()
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy())
        recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())
        
        # Log metrics
        self.log('test_acc', acc, on_step=True, on_epoch=True)
        self.log('test_precision', precision, on_step=True, on_epoch=True)
        self.log('test_recall', recall, on_step=True, on_epoch=True)
        self.log('test_f1', f1, on_step=True, on_epoch=True)
        
        # Save test set attention maps (only for the first batch)
        if batch_idx == 0 and protein_attention is not None and rna_attention is not None:
            self._plot_attention(
                protein_attention[0],
                f"Test Protein to RNA Attention (Batch {batch_idx})",
                self.attention_dir / f"test_protein_attention_batch_{batch_idx}.png"
            )
            self._plot_attention(
                rna_attention[0],
                f"Test RNA to Protein Attention (Batch {batch_idx})",
                self.attention_dir / f"test_rna_attention_batch_{batch_idx}.png"
            )
        
        return {'test_acc': acc, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

def train_classifier():
    try:
        # Set random seed for reproducibility
        pl.seed_everything(42)
        
        # Ensure all required directories exist
        print("Checking and creating necessary directories...")
        log_dir = Path('lightning_logs')
        attention_dir = Path("attention_maps")
        
        # Create directories if they don't exist
        log_dir.mkdir(exist_ok=True, parents=True)
        attention_dir.mkdir(exist_ok=True, parents=True)
        
        print("Starting to load dataset...")
        # Load dataset with smaller batch size and fewer workers
        dataset = RPIDataset(
            data_path='data/NPInter2.csv',
            batch_size=2,  # Reduce batch size to 2
            num_workers=2,  # Use 2 workers to balance loading speed and memory usage
            rna_col='RNA_aa_code',
            protein_col='target_aa_code',
            label_col='Y',
            padding=True,
            rna_max_length=220,  
            protein_max_length=500,  
            truncation=True,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        print("Setting up dataset...")
        # Setup dataset
        dataset.setup()
        
        print("Initializing model...")
        # Initialize model with simplified classifier
        model = ProteinRNALightningModule(
            output_dim=1280,  # ESM-2 output dimension
            hidden_dim=256,   # Simplified hidden layer dimension
            dropout=0.1
        )
        
        print("GPU availability check:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        
        print("Setting up callbacks...")
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            save_top_k=1,
            filename='best_model-{epoch:02d}-{val_f1:.4f}'
        )
        
        print("Setting up logger...")
        # Setup logger, save logs in DeepRPI directory
        logger = TensorBoardLogger('lightning_logs', name='protein_rna_classifier')
        
        print("Initializing trainer...")
        # Initialize trainer with fewer epochs and gradient accumulation
        # Set devices=[0] to use only the first GPU, avoiding NCCL communication issues with multiple GPUs
        trainer = pl.Trainer(
            max_epochs=15,  # Reduced from 10 to 5
            accelerator='gpu',  # Explicitly specify using GPU
            devices=[0],        # Use only the first GPU, avoid multi-GPU communication issues
            callbacks=[checkpoint_callback],
            logger=logger,
            log_every_n_steps=10,
            accumulate_grad_batches=4,  # Accumulate gradients to simulate larger batch size
            gradient_clip_val=1.0,  # Add gradient clipping
            limit_train_batches=1.0,  # Use all training data
            limit_val_batches=1.0,  # Use all validation data
            num_sanity_val_steps=1,  # Reduce integrity check steps
        )
        
        print("Starting model training...")
        # Train model
        try:
            trainer.fit(
                model,
                train_dataloaders=dataset.train_dataloader(),
                val_dataloaders=dataset.val_dataloader()
            )
            print("Training completed!")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            print("Attempting to continue with testing...")
        
        print("Starting model testing...")
        # Test model - check if checkpoint is generated
        try:
            checkpoint_path = checkpoint_callback.best_model_path
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Using best checkpoint for testing: {checkpoint_path}")
                trainer.test(
                    model,
                    dataloaders=dataset.test_dataloader(),
                    ckpt_path=checkpoint_path
                )
            else:
                print("Best checkpoint not found, testing with current model...")
                trainer.test(
                    model,
                    dataloaders=dataset.test_dataloader()
                )
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            print("Testing with current model...")
            trainer.test(
                model,
                dataloaders=dataset.test_dataloader()
            )
        
        print("Training completed!")
    except Exception as e:
        print(f"Critical error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_classifier() 