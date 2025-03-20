from typing import Tuple, Any, List
import esm
import torch
import torch.nn as nn
from torch import device

def load_esm_model()-> Tuple[nn.Module, esm.Alphabet]:
    """
    Load the ESM-1b model.
    :return: The ESM-1b model.
    """
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    print("Model loaded successfully.")
    return model, alphabet

class TokenESMEmbedding:
    """
    To generate protein embeddings using ESM-1b model.
    """
    def __init__(self, model, alphabet, device: device):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, seqs) -> tuple[Any, list[Any], Any]:
        """
        Generate embeddings for the given sequences. This step is done by a pretrained model.
        :param seqs: The sequences for which embeddings are to be generated.
        :return: The embeddings for the given sequences.
        """
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

