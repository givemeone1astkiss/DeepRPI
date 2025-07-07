#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py
DeepRPI train script
"""

import argparse
from pathlib import Path
from deeprpi.utils import train_classifier, set_seed

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Train the protein-RNA interaction classifier")
    
    # data parameters
    parser.add_argument("--data_path", type=str, default="data/NPInter2.csv",
                        help="Training data file path")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading threads")
    
    # training parameters
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Training maximum epochs")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()

def main():
    """
    Main function
    """
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Ensure output directory exists
    log_dir = Path("lightning_logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 50)
    print("DeepRPI protein-RNA interaction training script")
    print("=" * 50)
    print(f"Data file: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Training epochs: {args.max_epochs}")
    print(f"Hidden layer dimension: {args.hidden_dim}")
    print(f"Dropout rate: {args.dropout}")
    print("=" * 50)
    
    # Start training
    model, training_results = train_classifier(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    
    # Print training results
    if model is not None and "test_results" in training_results:
        print("\nTraining completed!")
        print(f"Best model saved path: {training_results['best_checkpoint']}")
        print("\nTest results:")
        for metric, value in training_results["test_results"][0].items():
            print(f"{metric}: {value:.4f}")
    else:
        print("\nAn error occurred during training:")
        if "error" in training_results:
            print(training_results["error"])

if __name__ == "__main__":
    main() 