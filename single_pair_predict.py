import torch
import pytorch_lightning as pl
from deeprpi.model.embedding import load_esm, load_rnabert, ESMEmbedding, RNABertEmbedding
from deeprpi.model.classifier import SimpleProteinRNAClassifier
from deeprpi.utils import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

"""
single_pair_predict.py
This script is used to predict the interaction between a single RNA-protein pair.
Designed for experimental biologists to input sequences and get prediction results with attention maps.
"""

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
            protein_dim=protein_dim,
            rna_dim=rna_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, protein_seqs, rna_seqs):
        # Get embeddings
        protein_embeddings, _, _ = self.esm_embedding(protein_seqs)
        rna_embeddings, _, _ = self.rna_embedding(rna_seqs)
        
        # Forward pass through classifier
        logits, protein_attention, rna_attention = self.classifier(protein_embeddings, rna_embeddings)
        return logits, protein_attention, rna_attention

def predict(protein_seq, rna_seq, checkpoint_path=None):
    """
    Predict the interaction between a single RNA-protein pair.
    
    Args:
        protein_seq (str): Protein sequence (amino acid sequence)
        rna_seq (str): RNA sequence (nucleotide sequence)
        checkpoint_path (str, optional): Model checkpoint path, use pretrained model by default
        
    Returns:
        dict: Contains prediction result, probability and attention matrices
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = ProteinRNALightningModule(output_dim=1280)
    
    # Load trained weights
    if checkpoint_path is None:
        checkpoint_path = 'lightning_logs/protein_rna_classifier/version_21/checkpoints/best_model-epoch=00-val_f1=0.5962.ckpt'
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
    
    # Encode sequences
    protein_encoded, _ = protein_tokenizer.encode([protein_seq])
    rna_encoded, _ = rna_tokenizer.encode([rna_seq])
    
    # Move input data to appropriate device
    protein_encoded = protein_encoded.to(device)
    rna_encoded = rna_encoded.to(device)
    
    # Predict interaction
    with torch.no_grad():
        try:
            logits, protein_attention, rna_attention = model(protein_encoded, rna_encoded)
            prob = torch.sigmoid(logits).squeeze().item()
            pred = 1 if prob > 0.5 else 0
            
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

def plot_attention(attention_weights, title, save_path):
    """
    Plot and save attention heatmap.
    
    Args:
        attention_weights: Attention weight matrix
        title: Chart title
        save_path: Save path
    """
    if attention_weights is None:
        print(f"Warning: No attention weights available for {title}")
        return
    
    # Get attention matrix shape
    attention_cpu = attention_weights.cpu().detach().numpy()
    shape = attention_cpu.shape
    
    # Plot heatmap
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

if __name__ == "__main__":
    # Example sequences
    protein_seq = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    rna_seq = "AUGGCCAUUGGAAAGGGUGCCCUCUUAUUAGCAGCUUGAGAAUUACUGUAAU"
    
    # Create output directory
    output_dir = Path("prediction_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Predict and get results
    result = predict(protein_seq, rna_seq)
    
    if result is not None:
        print("\nPrediction results:")
        print(f"Predicted class: {'Interaction' if result['prediction'] == 1 else 'No interaction'}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        # Save attention heatmaps
        if 'protein_attention' in result and result['protein_attention'] is not None:
            protein_attention_path = output_dir / "protein_to_rna_attention.png"
            plot_attention(
                result['protein_attention'],
                "Protein to RNA Attention",
                protein_attention_path
            )
            print(f"Protein to RNA attention map saved to: {protein_attention_path}")
            
        if 'rna_attention' in result and result['rna_attention'] is not None:
            rna_attention_path = output_dir / "rna_to_protein_attention.png"
            plot_attention(
                result['rna_attention'],
                "RNA to Protein Attention",
                rna_attention_path
            )
            print(f"RNA to protein attention map saved to: {rna_attention_path}")
    else:
        print("\nPrediction failed, please check if the model file path is correct.") 