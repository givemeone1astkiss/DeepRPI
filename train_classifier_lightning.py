import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from deeprpi.model.embedding import load_esm, load_rnabert, ESMEmbedding, RNABertEmbedding
from deeprpi.model.classifier import ProteinRNAClassifier
from deeprpi.utils import RPIDataset

class ProteinRNALightningModule(pl.LightningModule):
    def __init__(
        self,
        output_dim: int = 256,
        hidden_dims: list = [512, 256],
        dropout: float = 0.1,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize embedding models
        self.esm_model, self.esm_alphabet = load_esm(use_pooling=True, output_dim=output_dim)
        self.rnabert_model, self.rnabert_tokenizer = load_rnabert(use_pooling=True, output_dim=output_dim)
        
        # Initialize embedding generators
        self.esm_embedding = ESMEmbedding(
            model=self.esm_model,
            alphabet=self.esm_alphabet,
            device=self.device,
            use_pooling=True,
            output_dim=output_dim
        )
        
        self.rna_embedding = RNABertEmbedding(
            model=self.rnabert_model,
            tokenizer=self.rnabert_tokenizer,
            device=self.device,
            max_length=440,
            use_pooling=True,
            output_dim=output_dim
        )
        
        # Initialize classifier
        self.classifier = ProteinRNAClassifier(
            input_dim=output_dim * 2,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, protein_seqs, rna_seqs):
        # Get embeddings
        protein_embeddings, _, _ = self.esm_embedding(protein_seqs, pool_embeddings=True)
        rna_embeddings, _, _ = self.rna_embedding(rna_seqs, pool_embeddings=True)
        
        # Forward pass through classifier
        logits = self.classifier(protein_embeddings, rna_embeddings)
        return logits
    
    def _process_batch(self, batch):
       
        rna_seqs = batch[0] 
        protein_seqs = batch[2]  
        labels = batch[4]  
        
        
        if len(labels.shape) > 1:
            labels = labels[:, 0]
            
        return rna_seqs, protein_seqs, labels
    
    def training_step(self, batch, batch_idx):
        rna_seqs, protein_seqs, labels = self._process_batch(batch)
        logits = self(protein_seqs, rna_seqs)
        loss = self.criterion(logits.squeeze(), labels.float())
        
        # Calculate metrics
        preds = (torch.sigmoid(logits) > 0.5).int()
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        rna_seqs, protein_seqs, labels = self._process_batch(batch)
        logits = self(protein_seqs, rna_seqs)
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
        logits = self(protein_seqs, rna_seqs)
        
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
        
        return {'test_acc': acc, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def train_classifier():
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Load dataset with original batch size and workers
    dataset = RPIDataset(
        data_path='./data/NPInter5.csv',
        batch_size=32,  
        num_workers=4,  
        rna_col='RNA_aa_code',
        protein_col='target_aa_code',
        label_col='Y',
        padding=True,
        rna_max_length=440,  
        protein_max_length=1000,  
        truncation=True,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Setup dataset
    dataset.setup()
    
    # Initialize model with original dimensions
    model = ProteinRNALightningModule(
        output_dim=512,  
        hidden_dims=[1024, 512, 256],  
        dropout=0.1,
        learning_rate=1e-4
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        save_top_k=1,
        filename='best_model-{epoch:02d}-{val_f1:.4f}'
    )
    
    # Setup logger
    logger = TensorBoardLogger('lightning_logs', name='protein_rna_classifier')
    
    # Initialize trainer with original epochs
    trainer = pl.Trainer(
        max_epochs=10,  
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10
    )
    
    # Train model
    trainer.fit(
        model,
        train_dataloaders=dataset.train_dataloader(),
        val_dataloaders=dataset.val_dataloader()
    )
    
    # Test model
    trainer.test(
        model,
        dataloaders=dataset.test_dataloader(),
        ckpt_path='best'
    )

if __name__ == "__main__":
    train_classifier() 