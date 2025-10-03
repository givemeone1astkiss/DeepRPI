import torch
import torch.nn as nn
from deeprpi.model.attention import CrossAttention

class DimensionAlignment(nn.Module):
    """
    Dimension alignment layer for aligning embeddings of different dimensions to the same dimension
    """
    def __init__(self, input_dim: int, target_dim: int, dropout: float = 0.1):
        """
        Initialize dimension alignment layer
        
        Args:
            input_dim: Input dimension
            target_dim: Target dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        
        # Projection layer
        self.projection = nn.Linear(input_dim, target_dim)
        # Layer normalization
        self.norm = nn.LayerNorm(target_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Activation function
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Aligned tensor [batch_size, seq_len, target_dim]
        """
        # Ensure input is on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Project to target dimension
        x = self.projection(x)
        # Activation function
        x = self.activation(x)
        # Layer normalization
        x = self.norm(x)
        # Dropout
        x = self.dropout(x)
        
        return x

class AttentionPooling(nn.Module):
    """
    Attention pooling layer that learns how to aggregate sequence information
    """
    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Initialize attention pooling layer
        
        Args:
            input_dim: Input dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, 1),
            nn.Softmax(dim=1)  # Apply softmax along sequence dimension
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Pooled tensor [batch_size, input_dim]
        """
        # Ensure input is on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Calculate attention weights
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        
        # Apply attention weights
        pooled = torch.sum(attention_weights * x, dim=1)  # [batch_size, input_dim]
        
        return pooled

class SimpleProteinRNAClassifier(nn.Module):
    """
    A sequence-level protein-RNA classifier with cross-attention mechanism.
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
        
        # Dimension alignment: align RNA dimension to protein dimension
        self.rna_dim_align = DimensionAlignment(
            input_dim=rna_dim,
            target_dim=protein_dim,
            dropout=dropout
        )
        
        # Unified dimension
        unified_dim = protein_dim
        
        # Add cross-attention module
        self.cross_attention = CrossAttention(
            protein_dim=unified_dim,  # Unified protein dimension
            rna_dim=unified_dim,      # Unified RNA dimension
            num_heads=8,              # Number of attention heads
            dropout=dropout           # Dropout rate
        )
        
        # Attention pooling layers that learn how to aggregate sequence information
        self.protein_pool = AttentionPooling(
            input_dim=unified_dim,
            dropout=dropout
        )
        self.rna_pool = AttentionPooling(
            input_dim=unified_dim,
            dropout=dropout
        )
        
        # Create MLP (using unified dimension)
        self.mlp = nn.Sequential(
            nn.Linear(unified_dim + unified_dim, hidden_dim),  # 2 * unified_dim
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
        
        # Dimension alignment: align RNA embeddings to protein embedding dimension
        rna_embeddings_aligned = self.rna_dim_align(rna_embeddings)
        
        # Apply cross-attention, keeping sequence dimension information
        protein_attended, rna_attended, protein_attention_weights, rna_attention_weights = self.cross_attention(
            protein_embeddings,
            rna_embeddings_aligned
        )
            
        # Use attention pooling to aggregate sequence information
        protein_pooled = self.protein_pool(protein_attended)  # [batch_size, unified_dim]
        rna_pooled = self.rna_pool(rna_attended)  # [batch_size, unified_dim]
        
        # Concatenate pooled embeddings
        combined_embeddings = torch.cat([protein_pooled, rna_pooled], dim=1)
        
        # Pass through MLP
        logits = self.mlp(combined_embeddings)
        
        return logits, protein_attention_weights, rna_attention_weights 