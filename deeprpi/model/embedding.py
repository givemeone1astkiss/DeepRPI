from typing import Tuple, Any
import esm
import torch
import torch.nn as nn
from torch import device
from ..config import glob
from multimolecule import RnaBertConfig, RnaBertModel
from multimolecule.tokenisers.rna import RnaTokenizer


def load_esm() -> Tuple[nn.Module, esm.Alphabet]:
    """
    Load the ESM-1b model.
    
    :param output_dim: Dimension of the output embeddings. If None, same as model's embedding dimension.
    :return: The ESM-1b model and alphabet
    """
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    print("Model loaded successfully.")
    return model, alphabet

def load_rnabert() -> Tuple[nn.Module, RnaTokenizer]:
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
        for i, (_, seq_len) in enumerate(zip(results["contacts"], batch_lens)):
            attention_contacts.append(results["contacts"][i][:seq_len, :seq_len].to(self.device))
        
        # Get token-level embeddings and ensure they are on the correct device
        token_embeddings = results["representations"][33].to(self.device)
        
        # Always return token-level embeddings
        return token_embeddings, attention_contacts, batch_lens

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
            output_dim: Dimension of the output embeddings. If None, same as model's hidden size.
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