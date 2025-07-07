import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """
    Cross-attention module for calculating attention relationships between protein and RNA sequences.
    Can calculate bidirectional attention and generate attention heatmaps.
    """
    
    def __init__(self, protein_dim: int, rna_dim: int, num_heads: int = 8, dropout: float = 0.1, device: str = None):
        """
        Initialize the cross-attention module.
        
        Args:
            protein_dim: Dimension of protein embeddings
            rna_dim: Dimension of RNA embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
            device: Device type ('cuda' or 'cpu')
        """
        super().__init__()
        self.num_heads = num_heads
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure dimensions are divisible by num_heads
        self.protein_dim = protein_dim - (protein_dim % num_heads)
        self.rna_dim = rna_dim - (rna_dim % num_heads)
        
        # Protein to RNA attention
        self.protein_to_rna = nn.MultiheadAttention(
            embed_dim=self.protein_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=self.rna_dim,
            vdim=self.rna_dim
        ).to(self.device)
        
        # RNA to protein attention
        self.rna_to_protein = nn.MultiheadAttention(
            embed_dim=self.rna_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=self.protein_dim,
            vdim=self.protein_dim
        ).to(self.device)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.protein_dim).to(self.device)
        self.norm2 = nn.LayerNorm(self.rna_dim).to(self.device)
        
        # Dropout
        self.dropout = nn.Dropout(dropout).to(self.device)
        
        # Projection layers for dimension adjustment
        if protein_dim != self.protein_dim:
            self.protein_projection = nn.Linear(protein_dim, self.protein_dim).to(self.device)
        else:
            self.protein_projection = nn.Identity().to(self.device)
            
        if rna_dim != self.rna_dim:
            self.rna_projection = nn.Linear(rna_dim, self.rna_dim).to(self.device)
        else:
            self.rna_projection = nn.Identity().to(self.device)
        
    def forward(self, protein_embeddings: torch.Tensor, rna_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass, calculate bidirectional cross-attention.
        
        Args:
            protein_embeddings: Protein embeddings [batch_size, protein_len, protein_dim]
            rna_embeddings: RNA embeddings [batch_size, rna_len, rna_dim]
            
        Returns:
            tuple containing:
            - Updated protein embeddings
            - Updated RNA embeddings
            - Protein-to-RNA attention heatmap
            - RNA-to-protein attention heatmap
        """
        # Ensure all tensors are on the same device
        protein_embeddings = protein_embeddings.to(self.device)
        rna_embeddings = rna_embeddings.to(self.device)
        
        # Ensure input dimensions are correct
        if len(protein_embeddings.shape) == 2:
            protein_embeddings = protein_embeddings.unsqueeze(1)
        if len(rna_embeddings.shape) == 2:
            rna_embeddings = rna_embeddings.unsqueeze(1)
            
        # Project to correct dimensions
        protein_embeddings = self.protein_projection(protein_embeddings)
        rna_embeddings = self.rna_projection(rna_embeddings)
            
        # Calculate protein to RNA attention
        protein_attended, protein_attention_weights = self.protein_to_rna(
            query=protein_embeddings,
            key=rna_embeddings,
            value=rna_embeddings,
            need_weights=True
        )
        
        # Calculate RNA to protein attention
        rna_attended, rna_attention_weights = self.rna_to_protein(
            query=rna_embeddings,
            key=protein_embeddings,
            value=protein_embeddings,
            need_weights=True
        )
        
        # Residual connection and layer normalization
        protein_embeddings = self.norm1(protein_embeddings + self.dropout(protein_attended))
        rna_embeddings = self.norm2(rna_embeddings + self.dropout(rna_attended))
        
        return protein_embeddings, rna_embeddings, protein_attention_weights, rna_attention_weights 