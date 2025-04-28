from typing import Tuple, Any
import esm
import torch
import torch.nn as nn
from torch import device
from ..config import glob
from multimolecule import RnaBertConfig, RnaBertModel
from multimolecule.tokenisers.rna import RnaTokenizer


def load_esm(output_dim: int = None) -> Tuple[nn.Module, esm.Alphabet]:
    """
    Load the ESM-1b model.
    
    :param output_dim: Dimension of the output embeddings. If None, same as model's embedding dimension.
    :return: The ESM-1b model and alphabet
    """
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    print("Model loaded successfully.")
    return model, alphabet

def load_rnabert(output_dim: int = None) -> Tuple[nn.Module, RnaTokenizer]:
    """
    Load the RNA BERT model.
    
    :param output_dim: Dimension of the output embeddings. If None, same as model's hidden size.
    :return: The RNA BERT model and tokenizer
    """
    config = RnaBertConfig()
    model = RnaBertModel(config)
    tokenizer = RnaTokenizer()
    model.eval() 
    print("RNAbert model loaded successfully.")
    return model, tokenizer

class ProteinFeatureExtractor:
    """
    A feature extractor for protein sequences using ESM model.
    Can extract both token-level and sequence-level features.
    """

    def __init__(self, model = None, alphabet = None, device = None):
        """
        Initialize the feature extractor.

        :param model: Optional pre-loaded ESM model
        :param alphabet: Optional pre-loaded ESM alphabet
        :param device: Target device (CPU/GPU)
        """

        # Move model to target device and set to evaluation mode
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize batch converter for sequence processing
        if model is None or alphabet is None:
            self.model, alphabet = load_esm()
        else:
            self.model = model
            self.alphabet = alphabet

        # Move model to target device and set to evaluation mode
        self.model = self.model.to(self.device).eval()

        # Initialize batch converter for sequence processing
        self.batch_converter = self.alphabet.get_batch_converter()

        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def __call__(self, raw_seqs, output_lever = 'token'):
        """
        Extract features from protein sequences.

        :param raw_seqs: Input protein sequences (encoded)
        :param output_lever: 'token' for per-residue features or
                             'sequence' for pooled features
        :return: Tuple containing features and contact information
        """

        # Get start and end tokens for sequence processing
        start_token = glob.AMINO_ACIDS['<bos>']
        end_token = glob.AMINO_ACIDS['<eos>']
        idx_to_token = {v: k for k, v in glob.AMINO_ACIDS.items()}

        # Convert encoded sequences to amino acid strings
        seqs = []
        for seq in raw_seqs:
            # Locate start and end positions in sequence
            start_idx = list(seq).index(start_token) + 1
            end_idx = list(seq).index(end_token)
            # Convert to amino acid string
            seq_str = ''.join([idx_to_token[int(idx)] for idx in seq[start_idx:end_idx]])
            seqs.append(seq_str)

        # Prepare input data for the model
        data = [(f"protein{i}", seq) for i, seq in enumerate(seqs)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)


        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts = True)

        # Return features based on output level
        if output_lever == 'token':
            # Token-level feature extraction
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
            contact_maps = [results["contacts"][i][:l, :l] for i, l in enumerate(batch_lens)]
            return  results["representations"][33], contact_maps, batch_lens
        else:
            # Sequence-level feature extraction
            embeddings = results["representations"][33]
            contact_maps = results["contacts"]

            # Ensure consistent sequence length
            seq_len = min(embeddings.size(1), contact_maps.size(1))
            embeddings = embeddings[:, :seq_len, :]
            contact_maps = contact_maps[:, :seq_len, :seq_len]

            # Perform attention-based pooling using contact maps as weights
            weights = contact_maps.mean(dim=2)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
            pooled_emb = (embeddings * weights.unsqueeze(-1)).sum(dim=1)
            return pooled_emb, self._get_contact_stats(contact_maps), None


class ESMEmbedding:
    """
    To generate protein embeddings using ESM-1b model.
    """

    def __init__(self, model, alphabet, device: device, output_dim: int = None):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        
        # Create pooling layer but don't use it in __call__
        embedding_dim = model.embed_tokens.embedding_dim
        if output_dim is None:
            output_dim = embedding_dim
        self.pooling = ProteinEmbeddingPooling(embedding_dim=embedding_dim, output_dim=output_dim).to(device)

    def __call__(self, raw_seqs) -> tuple[Any, list[Any], Any]:
        """
        Generate embeddings for the given sequences. This step is done by a pretrained model.
        :param raw_seqs: The sequences for which embeddings are to be generated.
        :return: The token-level embeddings for the given sequences.
        """
        # Extract start and end token indices
        start_token = glob.AMINO_ACIDS['<bos>']
        end_token = glob.AMINO_ACIDS['<eos>']
        idx_to_token = {v: k for k, v in glob.AMINO_ACIDS.items()}

        # Convert sequences to strings and retain only the part between start and end tokens
        seqs = []
        for seq in raw_seqs:
            start_idx = list(seq).index(start_token) + 1
            end_idx = list(seq).index(end_token)
            seq_str = ''.join([idx_to_token[int(idx)] for idx in seq[start_idx:end_idx]])
            seqs.append(seq_str)

        # Existing code to generate embeddings and contacts
        data = [(f"protein{i}", seq) for i, seq in enumerate(seqs)]
        _, _, batch_tokens = self.batch_converter(data)
        # Ensure batch_tokens is on the correct device
        batch_tokens = batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        
        with torch.no_grad():
            # Ensure model is on the correct device
            self.model = self.model.to(self.device)
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
        
        attention_contacts = []
        # Extract the attention contacts for each sequence.
        for i, (contact, seq_len) in enumerate(zip(results["contacts"], batch_lens)):
            attention_contacts.append(results["contacts"][i][:seq_len, :seq_len].to(self.device))
        
        # Get token-level embeddings and ensure they are on the correct device
        token_embeddings = results["representations"][33].to(self.device)
        
        # Always return token-level embeddings
        return token_embeddings, attention_contacts, batch_lens

class RNABertEmbedding:
    """
    To generate RNA embeddings using a BERT model.
    """
    def __init__(self, model, tokenizer, device: device, max_length: int=440, output_dim: int = None):
        """
        Initialize RNABertEmbedding.
        
        Args:
            model: The RNAbert model.
            tokenizer: The RNAbert tokenizer.
            device: The device on which to run the model.
            max_length: Maximum sequence length for tokenization.
            output_dim: Dimension of the output embeddings. If None, same as model's hidden size.
        """
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create pooling layer but don't use it in __call__
        embedding_dim = model.config.hidden_size
        if output_dim is None:
            output_dim = embedding_dim
        self.pooling = ProteinEmbeddingPooling(embedding_dim=embedding_dim, output_dim=output_dim).to(device)

    def __call__(self, raw_seqs, return_attention: bool = False) -> tuple[Any, list[Any], Any]:
        """
        Generate embeddings for the given RNA sequences.
        
        Args:
            raw_seqs: The RNA sequences for which embeddings are to be generated.
                     Can be a list of strings or tokenized sequences.
            return_attention: Whether to return attention matrices.
        
        Returns:
            A tuple containing:
            - The token-level embeddings for the given sequences.
            - Attention matrices if return_attention=True, else None.
            - Sequence lengths.
        """
        start_token = glob.RNA_BASES['<bos>']
        end_token = glob.RNA_BASES['<eos>']
        idx_to_token = {v: k for k, v in glob.RNA_BASES.items()}
        seqs = []
        for seq in raw_seqs:
            start_idx = list(seq).index(start_token) + 1
            end_idx = list(seq).index(end_token)
            seq_str = ''.join([idx_to_token[int(idx)] for idx in seq[start_idx:end_idx]])
            seqs.append(seq_str)
        # Use tokenizer to process sequences
        inputs = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Calculate sequence lengths (excluding padding tokens)
        batch_lens = (inputs['attention_mask'] == 1).sum(1)
        
        # Get embedding vectors and optional attention matrices
        with torch.no_grad():
            # Call model with output_attentions if requested
            outputs = self.model(
                **inputs,
                output_attentions=return_attention
            )
            
            # Get token-level embeddings and ensure they are on the correct device
            token_embeddings = outputs["last_hidden_state"].to(self.device)
            
            # Always return token-level embeddings
            embeddings = token_embeddings
            
            # Process attention matrices if requested
            attention_matrices = None
            if return_attention:
                # Get attention from the last layer and ensure it's on the correct device
                last_layer_attention = outputs.attentions[-1].to(self.device)
                
                # Average across all attention heads
                attention_avg = last_layer_attention.mean(dim=1)
                
                # Extract attention matrices for each sequence based on its actual length
                attention_matrices = []
                for i, seq_len in enumerate(batch_lens):
                    seq_len = seq_len.item()
                    attention_matrices.append(attention_avg[i, :seq_len, :seq_len])
        
        return embeddings, attention_matrices, batch_lens

class ProteinEmbeddingPooling(nn.Module):
    """
    Advanced pooling class for converting token-level protein embeddings to sequence-level embeddings.
    Uses self-attention mechanism to capture global dependencies in the sequence.
    """
    
    def __init__(self, embedding_dim: int, output_dim: int = None, dropout: float = 0.1):
        """
        Initialize the protein embedding pooling class.
        
        Args:
            embedding_dim: Dimension of the input embedding vectors
            output_dim: Dimension of the output embeddings. If None, same as input dimension.
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim if output_dim is not None else embedding_dim
        self.dropout = nn.Dropout(dropout)
        
        # Self-attention pooling
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.scale = embedding_dim ** -0.5
        
        # Output projection layer for further processing of pooled representations
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Convert token-level embeddings to sequence-level embeddings.
        
        Args:
            embeddings: Embedding tensor of shape [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask of shape [batch_size, seq_len], 1 for valid tokens, 0 for padding tokens
            
        Returns:
            Sequence-level embeddings of shape [batch_size, output_dim]
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # If no attention mask is provided, assume all tokens are valid
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=embeddings.device)
        
        # Expand attention mask to match embedding dimensions
        attention_mask = attention_mask.unsqueeze(-1)
        
        # Self-attention pooling
        # Compute query, key, and value
        query = self.query(embeddings)
        key = self.key(embeddings)
        value = self.value(embeddings)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Set attention scores for padding positions to a very small value
        attention_scores = attention_scores.masked_fill(
            attention_mask.squeeze(-1).unsqueeze(1) == 0, -1e9
        )
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Weighted sum to get sequence representation
        context = torch.matmul(attention_weights, value)
        
        # Take the representation of the first token as the sequence representation
        # Other strategies like averaging or taking maximum can also be used
        sequence_embedding = context[:, 0, :]
        
        # Apply output projection
        sequence_embedding = self.output_projection(sequence_embedding)
        
        return sequence_embedding

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