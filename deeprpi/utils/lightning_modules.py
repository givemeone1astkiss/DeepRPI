import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from deeprpi.model.embedding import load_esm, load_rnabert, ESMEmbedding, RNABertEmbedding
from deeprpi.model.classifier import SimpleProteinRNAClassifier

class ProteinRNALightningModule(pl.LightningModule):
    """
    Lightning module for protein-RNA interaction classification.
    Encapsulates model, training, validation, and testing steps.
    """
    
    def __init__(self, output_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        """
        Initialize the lightning module.
        
        Args:
            output_dim: Output dimension for the classifier
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize embeddings
        self.esm_embedding = ESMEmbedding(*load_esm(), device=self.device)
        self.rna_embedding = RNABertEmbedding(*load_rnabert(), device=self.device)
        
        # ESM-2 output dimension is 1280, RNA-BERT output dimension is 120
        protein_dim = 1280
        rna_dim = 120
        
        # Initialize classifier with correct dimensions
        self.classifier = SimpleProteinRNAClassifier(
            protein_dim=protein_dim,
            rna_dim=rna_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Create attention map save directory
        self.attention_dir = Path("attention_maps")
        self.attention_dir.mkdir(exist_ok=True, parents=True)
        
    def forward(self, protein_seqs, rna_seqs):
        """
        Forward pass through the model.
        
        Args:
            protein_seqs: Protein sequences
            rna_seqs: RNA sequences
            
        Returns:
            tuple containing:
            - Logits [batch_size, 1]
            - Protein-to-RNA attention weights
            - RNA-to-protein attention weights
        """
        # Get embeddings - default no pooling, preserve sequence dimension
        protein_embeddings, _, _ = self.esm_embedding(protein_seqs)
        rna_embeddings, _, _ = self.rna_embedding(rna_seqs)
        
        # Forward pass through classifier
        # The classifier will handle cross-attention, pooling, concatenation and classification internally
        logits, protein_attention, rna_attention = self.classifier(protein_embeddings, rna_embeddings)
        return logits, protein_attention, rna_attention
    
    def _process_batch(self, batch):
        """
        Process a batch of data.
        
        Args:
            batch: Batch of data
            
        Returns:
            tuple containing:
            - RNA sequences
            - Protein sequences
            - Labels
        """
        # Handle different batch formats (dict vs list)
        if isinstance(batch, dict):
            rna_seqs = batch.get('rna_seq')
            protein_seqs = batch.get('protein_seq')
            labels = batch.get('label')
        else:
            rna_seqs = batch[0] 
            protein_seqs = batch[2]  
            labels = batch[4]
        
        # Ensure labels are correctly shaped
        if labels is not None and len(labels.shape) > 1:
            labels = labels[:, 0]
            
        return rna_seqs, protein_seqs, labels
    
    def _plot_attention(self, attention_weights, title, save_path):
        """
        Plot attention heatmap and save to file.
        
        Args:
            attention_weights: Attention weight matrix
            title: Title of the plot
            save_path: Path to save the plot
        """
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
        """
        Training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
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
        
        # Save attention maps (every 500 batches)
        if batch_idx % 500 == 0 and protein_attention is not None:
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
        """
        Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary containing validation metrics
        """
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
        """
        Test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary containing test metrics
        """
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
        """
        Configure optimizers.
        
        Returns:
            Optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer 