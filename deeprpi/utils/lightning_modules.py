import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
        
        # Initialize embeddings (device will be set by Lightning)
        self.esm_embedding = ESMEmbedding(*load_esm(), device='cpu')
        self.rna_embedding = RNABertEmbedding(*load_rnabert(), device='cpu')
        
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
        
        # Create CSV logs directory
        self.csv_logs_dir = Path("csv_logs")
        self.csv_logs_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize metrics storage for CSV
        self.training_metrics = []
        self.validation_metrics = []
        
        # Batch interval for saving metrics (save every 100 batches)
        self.save_interval = 100
        
        # Track current epoch metrics for memory efficiency
        self.current_epoch_training_metrics = []
        self.current_epoch_validation_metrics = []
        
        # Track metrics for batch averaging
        self.batch_metrics_buffer = []
        self.val_batch_metrics_buffer = []
        
        # Get process ID for distinguishing different processes
        import os
        self.process_id = os.getpid()
        
        # Track metrics for cross-process averaging
        self.epoch_training_metrics_buffer = []
        self.epoch_validation_metrics_buffer = []
    
    def on_train_start(self):
        """Called when training starts, ensures all components are on the correct device"""
        device = self.device
        # Move embedding models to the correct device
        self.esm_embedding.device = device
        self.esm_embedding.model = self.esm_embedding.model.to(device)
        self.rna_embedding.device = device
        self.rna_embedding.model = self.rna_embedding.model.to(device)
        print(f"Models moved to device: {device}")
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch to clear current epoch metrics"""
        self.current_epoch_training_metrics.clear()
        print(f"Starting training epoch {self.current_epoch}")
    
    def on_validation_epoch_start(self):
        """Called at the start of each validation epoch to clear current epoch metrics"""
        self.current_epoch_validation_metrics.clear()
        print(f"Starting validation epoch {self.current_epoch}")
    
    def on_validation_start(self):
        """Called when validation starts, ensures all components are on the correct device"""
        device = self.device
        self.esm_embedding.device = device
        self.esm_embedding.model = self.esm_embedding.model.to(device)
        self.rna_embedding.device = device
        self.rna_embedding.model = self.rna_embedding.model.to(device)
    
    def on_test_start(self):
        """Called when testing starts, ensures all components are on the correct device"""
        device = self.device
        self.esm_embedding.device = device
        self.esm_embedding.model = self.esm_embedding.model.to(device)
        self.rna_embedding.device = device
        self.rna_embedding.model = self.rna_embedding.model.to(device)
        
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
        # Ensure sequences are on the correct device
        device = self.device
        protein_seqs = protein_seqs.to(device)
        rna_seqs = rna_seqs.to(device)
        
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
        
        # Calculate additional metrics for CSV
        precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), zero_division=0)
        recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), zero_division=0)
        
        # Store metrics in buffer for averaging
        self.batch_metrics_buffer.append({
            'loss': loss.item(),
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
        
        # Save averaged metrics every save_interval batches
        if (batch_idx + 1) % self.save_interval == 0:
            epoch = self.current_epoch
            step = self.global_step
            
            # Calculate average metrics over the last save_interval batches
            avg_metrics = {
                'epoch': epoch,
                'step': step,
                'batch_idx': batch_idx,
                'process_id': self.process_id,
                'loss': np.mean([m['loss'] for m in self.batch_metrics_buffer]),
                'accuracy': np.mean([m['accuracy'] for m in self.batch_metrics_buffer]),
                'f1': np.mean([m['f1'] for m in self.batch_metrics_buffer]),
                'precision': np.mean([m['precision'] for m in self.batch_metrics_buffer]),
                'recall': np.mean([m['recall'] for m in self.batch_metrics_buffer])
            }
            
            # Append to CSV file directly
            csv_file = self.csv_logs_dir / 'training_metrics.csv'
            if csv_file.exists():
                # Append to existing file
                df_new = pd.DataFrame([avg_metrics])
                df_new.to_csv(csv_file, mode='a', header=False, index=False)
            else:
                # Create new file with header
                df_new = pd.DataFrame([avg_metrics])
                df_new.to_csv(csv_file, index=False)
            
            # Also store in memory for epoch-level aggregation
            self.training_metrics.append(avg_metrics)
            self.current_epoch_training_metrics.append(avg_metrics)
            self.epoch_training_metrics_buffer.append(avg_metrics)
            
            # Clear buffer for next interval
            self.batch_metrics_buffer.clear()
        
        # Save attention maps (every 500 batches)
        if batch_idx % 500 == 0 and protein_attention is not None:
            epoch = self.current_epoch
            self._plot_attention(
                protein_attention[0],  # Take attention of the first sample
                f"Protein to RNA Attention (Epoch {epoch}, Batch {batch_idx})",
                self.attention_dir / f"protein_attention_epoch_{epoch}_batch_{batch_idx}.png"
            )
            self._plot_attention(
                rna_attention[0],  # Take attention of the first sample
                f"RNA to Protein Attention (Epoch {epoch}, Batch {batch_idx})",
                self.attention_dir / f"rna_attention_epoch_{epoch}_batch_{batch_idx}.png"
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
        
        # Calculate additional metrics for CSV
        precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), zero_division=0)
        recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), zero_division=0)
        
        # Store metrics in buffer for averaging
        self.val_batch_metrics_buffer.append({
            'loss': loss.item(),
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
        
        # Save averaged metrics every save_interval batches
        if (batch_idx + 1) % self.save_interval == 0:
            epoch = self.current_epoch
            step = self.global_step
            
            # Calculate average metrics over the last save_interval batches
            avg_metrics = {
                'epoch': epoch,
                'step': step,
                'batch_idx': f"val_{batch_idx}",
                'process_id': self.process_id,
                'loss': np.mean([m['loss'] for m in self.val_batch_metrics_buffer]),
                'accuracy': np.mean([m['accuracy'] for m in self.val_batch_metrics_buffer]),
                'f1': np.mean([m['f1'] for m in self.val_batch_metrics_buffer]),
                'precision': np.mean([m['precision'] for m in self.val_batch_metrics_buffer]),
                'recall': np.mean([m['recall'] for m in self.val_batch_metrics_buffer])
            }
            
            # Append to CSV file directly (more efficient than rewriting entire file)
            csv_file = self.csv_logs_dir / 'validation_metrics.csv'
            if csv_file.exists():
                # Append to existing file
                df_new = pd.DataFrame([avg_metrics])
                df_new.to_csv(csv_file, mode='a', header=False, index=False)
            else:
                # Create new file with header
                df_new = pd.DataFrame([avg_metrics])
                df_new.to_csv(csv_file, index=False)
            
            # Also store in memory for epoch-level aggregation
            self.validation_metrics.append(avg_metrics)
            self.current_epoch_validation_metrics.append(avg_metrics)
            self.epoch_validation_metrics_buffer.append(avg_metrics)
            
            # Clear buffer for next interval
            self.val_batch_metrics_buffer.clear()
        
        return {'val_loss': loss, 'val_acc': acc, 'val_f1': f1}
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch to save epoch-level aggregated metrics"""
        print(f"on_train_epoch_end called for epoch {self.current_epoch}")
        
        if self.epoch_training_metrics_buffer:
            try:
                # Gather metrics from all processes
                gathered_metrics = self.all_gather(self.epoch_training_metrics_buffer)
                
                # Flatten the gathered metrics (it comes as a list of lists)
                all_metrics = []
                for process_metrics in gathered_metrics:
                    all_metrics.extend(process_metrics)
                
                if all_metrics:
                    # Calculate cross-process average
                    df_all = pd.DataFrame(all_metrics)
                    current_epoch = self.current_epoch
                    
                    print(f"Found {len(df_all)} training metrics across all processes for epoch {current_epoch}")
                    
                    # Calculate mean metrics across all processes
                    epoch_summary = {
                        'epoch': current_epoch,
                        'loss': df_all['loss'].mean(),
                        'accuracy': df_all['accuracy'].mean(),
                        'f1': df_all['f1'].mean(),
                        'precision': df_all['precision'].mean(),
                        'recall': df_all['recall'].mean()
                    }
                    
                    print(f"Epoch {current_epoch} cross-process training summary: {epoch_summary}")
                    
                    # Only save from main process to avoid duplicate files
                    if self.trainer.is_global_zero:
                        # Append to epoch-level CSV (create if doesn't exist)
                        epoch_file = self.csv_logs_dir / 'training_epoch_metrics.csv'
                        if epoch_file.exists():
                            df_epoch = pd.read_csv(epoch_file)
                            df_epoch = pd.concat([df_epoch, pd.DataFrame([epoch_summary])], ignore_index=True)
                        else:
                            df_epoch = pd.DataFrame([epoch_summary])
                        
                        df_epoch.to_csv(epoch_file, index=False)
                        print(f"Saved epoch {current_epoch} cross-process training metrics to {epoch_file}")
                
                # Clear current epoch metrics to free memory
                self.current_epoch_training_metrics.clear()
                self.epoch_training_metrics_buffer.clear()
                
            except Exception as e:
                print(f"Error in on_train_epoch_end: {e}")
        else:
            print("No training metrics available for current epoch")
            
    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch to save epoch-level aggregated metrics"""
        print(f"on_validation_epoch_end called for epoch {self.current_epoch}")
        
        if self.epoch_validation_metrics_buffer:
            try:
                # Gather metrics from all processes
                gathered_metrics = self.all_gather(self.epoch_validation_metrics_buffer)
                
                # Flatten the gathered metrics (it comes as a list of lists)
                all_metrics = []
                for process_metrics in gathered_metrics:
                    all_metrics.extend(process_metrics)
                
                if all_metrics:
                    # Calculate cross-process average
                    df_all = pd.DataFrame(all_metrics)
                    current_epoch = self.current_epoch
                    
                    print(f"Found {len(df_all)} validation metrics across all processes for epoch {current_epoch}")
                    
                    # Calculate mean metrics across all processes
                    epoch_summary = {
                        'epoch': current_epoch,
                        'loss': df_all['loss'].mean(),
                        'accuracy': df_all['accuracy'].mean(),
                        'f1': df_all['f1'].mean(),
                        'precision': df_all['precision'].mean(),
                        'recall': df_all['recall'].mean()
                    }
                    
                    print(f"Epoch {current_epoch} cross-process validation summary: {epoch_summary}")
                    
                    # Only save from main process to avoid duplicate files
                    if self.trainer.is_global_zero:
                        # Append to epoch-level CSV (create if doesn't exist)
                        epoch_file = self.csv_logs_dir / 'validation_epoch_metrics.csv'
                        if epoch_file.exists():
                            df_epoch = pd.read_csv(epoch_file)
                            df_epoch = pd.concat([df_epoch, pd.DataFrame([epoch_summary])], ignore_index=True)
                        else:
                            df_epoch = pd.DataFrame([epoch_summary])
                        
                        df_epoch.to_csv(epoch_file, index=False)
                        print(f"Saved epoch {current_epoch} cross-process validation metrics to {epoch_file}")
                
                # Clear current epoch metrics to free memory
                self.current_epoch_validation_metrics.clear()
                self.epoch_validation_metrics_buffer.clear()
                
            except Exception as e:
                print(f"Error in on_validation_epoch_end: {e}")
        else:
            print("No validation metrics available for current epoch")
    
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
            # Use a timestamp or test identifier instead of epoch for test
            import time
            test_id = int(time.time())
            self._plot_attention(
                protein_attention[0],
                f"Test Protein to RNA Attention (Test ID: {test_id}, Batch {batch_idx})",
                self.attention_dir / f"test_protein_attention_id_{test_id}_batch_{batch_idx}.png"
            )
            self._plot_attention(
                rna_attention[0],
                f"Test RNA to Protein Attention (Test ID: {test_id}, Batch {batch_idx})",
                self.attention_dir / f"test_rna_attention_id_{test_id}_batch_{batch_idx}.png"
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