from typing import Tuple, Any
import esm
import torch
import torch.nn as nn
from torch import device
from ..config import glob


def load_esm() -> Tuple[nn.Module, esm.Alphabet]:
    """
    Load the ESM-1b model.
    :return: The ESM-1b model.
    """
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    print("Model loaded successfully.")
    return model, alphabet

def load_rnabert() -> None:
    """
    Load the RNA BERT model.
    :return: The RNA BERT model.
    """
    pass

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
    def __init__(self):
        pass

    def __call__(self, raw_seqs):
        pass
