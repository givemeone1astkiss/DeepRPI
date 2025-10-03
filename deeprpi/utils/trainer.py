import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import os

from deeprpi.utils.lightning_modules import ProteinRNALightningModule
from deeprpi.utils.data import RPIDataset

def train_classifier(
    data_path='data/NPInter2.csv',
    batch_size=8,
    num_workers=4,
    max_epochs=10,
    output_dim=1280,
    hidden_dim=256,
    dropout=0.1,
    model_seed=42,
    data_split_seed=42
):
    """
    Train a protein-RNA classifier using PyTorch Lightning.
    
    Args:
        data_path: Path to the data file
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        max_epochs: Maximum number of epochs for training
        output_dim: Output dimension for the classifier
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
        model_seed: Random seed for model initialization and training
        data_split_seed: Random seed for data splitting (should be fixed for reproducibility)
        
    Returns:
        Trained model and training results
    """
    try:
        # Set random seed for model initialization and training only
        pl.seed_everything(model_seed)
        
        # Ensure all required directories exist
        print("Checking and creating necessary directories...")
        log_dir = Path('lightning_logs')
        attention_dir = Path("attention_maps")
        
        # Create directories if they don't exist
        log_dir.mkdir(exist_ok=True, parents=True)
        attention_dir.mkdir(exist_ok=True, parents=True)
        
        print("Starting to load dataset...")
        # Load dataset with fixed data split seed
        dataset = RPIDataset(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            rna_col='RNA_aa_code',
            protein_col='target_aa_code',
            label_col='Y',
            padding=True,
            rna_max_length=220,
            protein_max_length=500,
            truncation=True,
            val_ratio=0.1,
            test_ratio=0.1,
            data_split_seed=data_split_seed
        )
        
        print("Setting up dataset...")
        # Setup dataset
        dataset.setup()
        
        print("Initializing model...")
        # Initialize model
        model = ProteinRNALightningModule(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
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
        # Setup logger
        logger = TensorBoardLogger('lightning_logs', name='protein_rna_classifier')
        
        print("Initializing trainer...")
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices='auto' if torch.cuda.is_available() else None,
            callbacks=[checkpoint_callback],
            logger=logger,
            log_every_n_steps=50,
            accumulate_grad_batches=4,
            gradient_clip_val=1.0,
            limit_train_batches=1.0,
            limit_val_batches=0.5,
            limit_test_batches=0.5,
            num_sanity_val_steps=0,
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
                test_results = trainer.test(
                    model,
                    dataloaders=dataset.test_dataloader(),
                    ckpt_path=checkpoint_path
                )
            else:
                print("Best checkpoint not found, testing with current model...")
                test_results = trainer.test(
                    model,
                    dataloaders=dataset.test_dataloader()
                )
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            print("Testing with current model...")
            test_results = trainer.test(
                model,
                dataloaders=dataset.test_dataloader()
            )
        
        print("Training and testing completed!")
        return model, {
            'best_checkpoint': checkpoint_callback.best_model_path if hasattr(checkpoint_callback, 'best_model_path') else None,
            'test_results': test_results
        }
    except Exception as e:
        print(f"Critical error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, {'error': str(e)} 