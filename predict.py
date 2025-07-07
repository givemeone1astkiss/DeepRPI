#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predict.py
Predict single protein-RNA interaction entry script
"""

import argparse
from pathlib import Path
from deeprpi.utils import predict_interaction

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Predict protein-RNA interaction")
    
    # input sequence
    parser.add_argument("--protein", type=str, required=True,
                        help="Protein sequence (amino acid sequence)")
    parser.add_argument("--rna", type=str, required=True,
                        help="RNA sequence (nucleotide sequence)")
    
    # 模型参数
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path, default using pre-trained model")
    
    # output parameters
    parser.add_argument("--output_dir", type=str, default="prediction_results",
                        help="Output directory")
    parser.add_argument("--plot_attention", action="store_true", default=True,
                        help="Whether to plot attention heatmap")
    
    return parser.parse_args()

def main():
    """
    Main function
    """
    args = parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 50)
    print("DeepRPI protein-RNA interaction prediction script")
    print("=" * 50)
    print(f"Protein sequence: {args.protein[:50]}..." if len(args.protein) > 50 else f"Protein sequence: {args.protein}")
    print(f"RNA sequence: {args.rna[:50]}..." if len(args.rna) > 50 else f"RNA sequence: {args.rna}")
    print(f"Model checkpoint: {args.checkpoint or 'Default'}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Predict interaction
    result = predict_interaction(
        protein_seq=args.protein,
        rna_seq=args.rna,
        checkpoint_path=args.checkpoint,
        plot_attention=args.plot_attention,
        output_dir=args.output_dir
    )
    
    # Print prediction results
    if result is not None:
        print("\nPrediction results:")
        print(f"Interaction prediction: {'exists' if result['prediction'] == 1 else 'does not exist'}")
        print(f"Interaction probability: {result['probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        if args.plot_attention:
            print(f"\nAttention heatmap saved to: {args.output_dir}")
    else:
        print("\nAn error occurred during prediction, unable to generate results")

if __name__ == "__main__":
    main() 