from typing import Tuple, Any, Dict, List, Union
import esm
import torch
import torch.nn as nn
from torch import device, Tensor
import numpy as np
from ..config import glob
from multimolecule import RnaBertConfig, RnaBertModel, RnaTokenizer

def load_esm() -> Tuple[nn.Module, esm.Alphabet]:
    """
    Load the ESM-1b model.
    :return: The ESM-1b model.
    """
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    print("Model loaded successfully.")
    return model, alphabet

def load_rnabert() -> Tuple[nn.Module, RnaTokenizer]:
    """
    Load the RNA BERT model.
    :return: The RNA BERT model and tokenizer.
    """
    config = RnaBertConfig()
    model = RnaBertModel(config)
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
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

    def __init__(self, model, alphabet, device: device):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, raw_seqs) -> tuple[Any, list[Any], Any]:
        """
        Generate embeddings for the given sequences. This step is done by a pretrained model.
        :param raw_seqs: The sequences for which embeddings are to be generated.
        :return: The embeddings for the given sequences.
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
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
        attention_contacts = []
        # Extract the attention contacts for each sequence.
        for i, (contact, seq_len) in enumerate(zip(results["contacts"], batch_lens)):
            attention_contacts.append(results["contacts"][i][:seq_len, :seq_len])
        return results["representations"][33], attention_contacts, batch_lens

class RNABertEmbedding:
    """
    To generate RNA embeddings using a BERT model.
    """
    def __init__(self, model, tokenizer, device: device, max_length: int=440):
        """
        Initialize RNABertEmbedding.
        
        Args:
            model: The RNAbert model.
            tokenizer: The RNAbert tokenizer.
            device: The device on which to run the model.
            max_length: Maximum sequence length for tokenization.
        """
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, raw_seqs, return_attention: bool = False) -> tuple[Any, list[Any], Any]:
        """
        Generate embeddings for the given RNA sequences.
        
        Args:
            raw_seqs: The RNA sequences for which embeddings are to be generated.
                     Can be a list of strings or tokenized sequences.
            return_attention: Whether to return attention matrices.
        
        Returns:
            A tuple containing:
            - The embeddings for the given sequences.
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
            
            # Get embedding vectors
            embeddings = outputs["pooler_output"]
            
            # Process attention matrices if requested
            attention_matrices = None
            if return_attention:
                # Get attention from the last layer
                last_layer_attention = outputs.attentions[-1]
                
                # Average across all attention heads
                attention_avg = last_layer_attention.mean(dim=1)
                
                # Extract attention matrices for each sequence based on its actual length
                attention_matrices = []
                for i, seq_len in enumerate(batch_lens):
                    seq_len = seq_len.item()
                    attention_matrices.append(attention_avg[i, :seq_len, :seq_len])
        
        return embeddings, attention_matrices, batch_lens
