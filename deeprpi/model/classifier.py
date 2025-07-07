import torch
import torch.nn as nn
from deeprpi.model.attention import CrossAttention

class SimpleProteinRNAClassifier(nn.Module):
    """
    A sequence-level protein-RNA classifier with cross-attention mechanism.
    Process flow: embedding → cross-attention → pooling → concatenation → classification
    """
    
    def __init__(self, protein_dim: int, rna_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        """
        Initialize the classifier.
        
        Args:
            protein_dim: Dimension of protein embeddings
            rna_dim: Dimension of RNA embeddings
            hidden_dim: Dimension of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Add cross-attention module
        self.cross_attention = CrossAttention(
            protein_dim=protein_dim,  # Protein embedding dimension
            rna_dim=rna_dim,         # RNA embedding dimension
            num_heads=8,             # Number of attention heads
            dropout=dropout          # Dropout rate
        )
        
        # Pooling layer to convert sequence-level representations to vectors
        self.protein_pool = nn.AdaptiveAvgPool1d(1)
        self.rna_pool = nn.AdaptiveAvgPool1d(1)
        
        # Create MLP
        self.mlp = nn.Sequential(
            nn.Linear(protein_dim + rna_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, protein_embeddings: torch.Tensor, rna_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            protein_embeddings: Protein sequence embeddings [batch_size, protein_dim] or [batch_size, seq_len, protein_dim]
            rna_embeddings: RNA sequence embeddings [batch_size, rna_dim] or [batch_size, seq_len, rna_dim]
            
        Returns:
            tuple containing:
            - Classification logits [batch_size, 1]
            - Protein-to-RNA attention heatmap
            - RNA-to-protein attention heatmap
        """
        # Ensure inputs have sequence dimension (if not, add a dummy sequence dimension)
        if len(protein_embeddings.shape) == 2:
            protein_embeddings = protein_embeddings.unsqueeze(1)
            
        if len(rna_embeddings.shape) == 2:
            rna_embeddings = rna_embeddings.unsqueeze(1)
            
        # Ensure both embeddings are on the same device
        device = protein_embeddings.device
        rna_embeddings = rna_embeddings.to(device)
        
        # Apply cross-attention, keeping sequence dimension information
        protein_attended, rna_attended, protein_attention_weights, rna_attention_weights = self.cross_attention(
            protein_embeddings,
            rna_embeddings
        )
            
        # Pool over sequence dimension to get fixed-dimension vectors
        # Transform dimensions to fit pooling layer input requirements: [batch, seq_len, dim] → [batch, dim, seq_len]
        protein_pooled = self.protein_pool(protein_attended.transpose(1, 2)).squeeze(-1)
        rna_pooled = self.rna_pool(rna_attended.transpose(1, 2)).squeeze(-1)
        
        # Concatenate pooled embeddings
        combined_embeddings = torch.cat([protein_pooled, rna_pooled], dim=1)
        
        # Pass through MLP
        logits = self.mlp(combined_embeddings)
        
        return logits, protein_attention_weights, rna_attention_weights 