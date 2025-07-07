import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from deeprpi.utils.lightning_modules import ProteinRNALightningModule
from deeprpi.utils.data import Tokenizer

def predict_interaction(protein_seq, rna_seq, checkpoint_path=None, plot_attention=True, output_dir=None):
    """
    Predict the interaction between a single RNA-protein pair.
    
    Args:
        protein_seq (str): Protein sequence (amino acid sequence)
        rna_seq (str): RNA sequence (nucleotide sequence)
        checkpoint_path (str, optional): Model checkpoint path, use pretrained model by default
        plot_attention (bool): Whether to plot and save attention maps
        output_dir (str, optional): Directory to save attention plots
        
    Returns:
        dict: Contains prediction result, probability and attention matrices
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set output directory for attention plots
    if plot_attention:
        output_dir = Path(output_dir or "attention_maps")
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    model = ProteinRNALightningModule(output_dim=1280)
    
    # Load trained weights
    if checkpoint_path is None:
        # Default checkpoint path
        checkpoint_path = 'DeepRPI/lightning_logs/protein_rna_classifier/version_1/checkpoints/best_model-epoch=01-val_f1=0.7697.ckpt'
    
    print(f"Using model file: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
    
    # Process sequences with Tokenizer
    protein_tokenizer = Tokenizer.protein(padding=True, max_length=500, to_tensor=True)
    rna_tokenizer = Tokenizer.rna(padding=True, max_length=220, to_tensor=True)
    
    # Encode sequences and ensure correct dimensions
    protein_encoded, _ = protein_tokenizer.encode([protein_seq])
    rna_encoded, _ = rna_tokenizer.encode([rna_seq])
    
    # Add batch dimension if not present
    if protein_encoded.dim() == 1:
        protein_encoded = protein_encoded.unsqueeze(0)
    if rna_encoded.dim() == 1:
        rna_encoded = rna_encoded.unsqueeze(0)
    
    # Move input data to appropriate device
    protein_encoded = protein_encoded.to(device)
    rna_encoded = rna_encoded.to(device)
    
    # Print tensor shapes for debugging
    print(f"Protein tensor shape: {protein_encoded.shape}")
    print(f"RNA tensor shape: {rna_encoded.shape}")
    
    # Predict interaction
    with torch.no_grad():
        try:
            logits, protein_attention, rna_attention = model(protein_encoded, rna_encoded)
            # Ensure logits is a 1D tensor
            if logits.dim() > 1:
                logits = logits.squeeze()
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob > 0.5 else 0
            
            # Plot attention maps if requested
            if plot_attention and protein_attention is not None and rna_attention is not None:
                try:
                    _plot_attention(
                        protein_attention[0], 
                        "Protein to RNA Attention", 
                        output_dir / "protein_to_rna_attention.png"
                    )
                    _plot_attention(
                        rna_attention[0], 
                        "RNA to Protein Attention", 
                        output_dir / "rna_to_protein_attention.png"
                    )
                except Exception as e:
                    print(f"Error plotting attention: {str(e)}")
            
            print("Prediction successful!")
            return {
                'prediction': pred,
                'probability': float(prob),
                'confidence': float(max(prob, 1-prob)),
                'protein_attention': protein_attention,
                'rna_attention': rna_attention
            }
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

def _plot_attention(attention_weights, title, save_path):
    """
    Plot attention heatmap and save.
    
    Args:
        attention_weights: Attention weight matrix
        title: Title for the plot
        save_path: Path to save the plot
    """
    # Get attention matrix shape
    attention_cpu = attention_weights.cpu().detach().numpy()
    shape = attention_cpu.shape
    
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
    plt.savefig(save_path, dpi=300)
    plt.close() 