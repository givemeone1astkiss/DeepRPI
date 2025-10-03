#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate.py
Evaluate the performance of DeepRPI model on test/validation set
"""

import argparse
from pathlib import Path
import pandas as pd
from deeprpi.utils import evaluate_dataset

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate protein-RNA interaction model performance")
    
    # data parameters
    parser.add_argument("--data_path", type=str, default="data/NPInter2.csv",
                        help="Data file path")
    
    # evaluation parameters
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Output directory")
    parser.add_argument("--save_attention", action="store_true", default=True,
                        help="Whether to save attention heatmap")
    parser.add_argument("--eval_val", action="store_true", default=False,
                        help="Evaluate validation set (default evaluate test set)")
    
    # seed parameters
    parser.add_argument("--data_split_seed", type=int, default=42,
                        help="Random seed for data splitting (must match training)")
    
    return parser.parse_args()

def main():
    """
    Main function
    """
    args = parse_args()
    
    # Note: No global seed setting here to avoid affecting data splitting
    # Data splitting seed is controlled in RPIDataset and must match training
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    set_type = "Validation set" if args.eval_val else "Test set"
    
    print("=" * 50)
    print(f"DeepRPI protein-RNA interaction model {set_type} evaluation script")
    print("=" * 50)
    print(f"Data file: {args.data_path}")
    print(f"Model checkpoint: {args.checkpoint or 'Default'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data split seed: {args.data_split_seed}")
    print("=" * 50)
    
    # Evaluate dataset
    metrics = evaluate_dataset(
        data_file=args.data_path,
        output_dir=args.output_dir,
        is_val=args.eval_val,
        save_attention=args.save_attention,
        checkpoint_path=args.checkpoint,
        data_split_seed=args.data_split_seed
    )
    
    # print evaluation results
    if metrics:
        print(f"\n{set_type} evaluation completed!")
        print("\nPerformance metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_csv = output_dir / f"{set_type.replace('set', '')}_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"\nMetrics saved to: {metrics_csv}")
        
        print(f"\nDetailed report saved to: {args.output_dir}")
    else:
        print("\nAn error occurred during evaluation, unable to generate results")

if __name__ == "__main__":
    main() 