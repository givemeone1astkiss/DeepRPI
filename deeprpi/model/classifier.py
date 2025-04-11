import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinRNAClassifier(nn.Module):
    """
    MLP classifier for protein-RNA interaction prediction.
    Takes concatenated protein and RNA embeddings as input and outputs binary classification.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [1024, 512, 256], dropout: float = 0.1):
        """
        Initialize the classifier.
        
        Args:
            input_dim: Dimension of the input embeddings (protein_dim + rna_dim)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        # Create MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Add final classification layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, protein_embeddings: torch.Tensor, rna_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.
        
        Args:
            protein_embeddings: Protein sequence embeddings [batch_size, protein_dim]
            rna_embeddings: RNA sequence embeddings [batch_size, rna_dim]
            
        Returns:
            Binary classification logits [batch_size, 1]
        """
        # Concatenate embeddings
        combined_embeddings = torch.cat([protein_embeddings, rna_embeddings], dim=1)
        
        # Pass through MLP
        logits = self.mlp(combined_embeddings)
        
        return logits 