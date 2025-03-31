import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Union
from numpy import ndarray
from deeprpi.config.glob import RNA_BASES, AMINO_ACIDS
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from ..config.glob import OUTPUT_PATH

def read_csv(file_path: str,
             rna_col: str,
             protein_col: str,
             label_col: str,) -> tuple[Any, Any, Any]:
    """
    Read RNA sequences, protein sequences and binding labels from CSV file.

    Args:
        file_path: Path to the CSV file containing the data
        rna_col: Name of the column containing RNA sequences
        protein_col: Name of the column containing protein sequences
        label_col: Name of the column containing binding labels

    Returns:
        Tuple containing:
        - List of RNA sequences
        - List of protein sequences
        - List of binding labels
    """
    df = pd.read_csv(file_path)

    # Extract RNA sequences, protein sequences and labels
    rna_seq = df[rna_col].values
    protein_seq = df[protein_col].values
    labels = df[label_col].values.tolist()

    return rna_seq, protein_seq, labels


class Tokenizer:
    @classmethod
    def rna(cls,
            padding: bool,
            max_length: int,
            to_tensor: bool):
        return cls(vocab=RNA_BASES, tokenizer_type="RNA", padding=padding, max_length=max_length, to_tensor=to_tensor)

    @classmethod
    def protein(cls,
                padding: bool,
                max_length: int,
                to_tensor: bool):
        return cls(vocab=AMINO_ACIDS, tokenizer_type="protein", padding=padding, max_length=max_length, to_tensor=to_tensor)

    def __init__(self, vocab: dict,
                 tokenizer_type: str,
                 padding: bool,
                 max_length: int,
                 to_tensor: bool):
        self.vocab = vocab
        self.type = tokenizer_type
        self.padding = padding
        self.max_length = max_length
        self.to_tensor = to_tensor

    @staticmethod
    def _tokenize(seq: str) -> List[str]:
        return [token for token in seq.upper()]

    def encode(self, seqs: List[str]) -> Union[Tuple[ndarray, ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        encoded = []
        for seq in tqdm(seqs, desc=f"Tokenizing {self.type} sequences"):
            if len(seq) > self.max_length-2:
                raise ValueError(f"Sequence length of {seq} is greater than the maximum length of {self.max_length}-2")
            seq = ["<bos>"]+self._tokenize(seq)+["<eos>"]
            encoded_seq = [self.vocab.get(token, self.vocab["<unk>"]) for token in seq]
            encoded.append(encoded_seq)

        padded = []
        masks = []
        if self.padding:
            for seq in tqdm(encoded, desc=f"Padding {self.type} sequences"):
                seq += [self.vocab["<pad>"]] * (self.max_length - len(seq))
                padded.append(seq)
                masks.append([1] * len(seq) + [0] * (self.max_length - len(seq)))
        else:
            padded = encoded
            masks = [[1] * len(seq) for seq in encoded]

        # Tokenize sequences.
        return (
            torch.tensor(padded, dtype=torch.long).squeeze(),
            torch.tensor(masks, dtype=torch.long).squeeze(),
        )

    def decode(self, seqs: torch.Tensor) -> List[str]:
        decoded = []
        for seq in seqs:
            decoded_seq = []
            for token in seq:
                if list(self.vocab.keys())[list(self.vocab.values()).index(token)] == "<bos>":
                    continue
                elif list(self.vocab.keys())[list(self.vocab.values()).index(token)] == "<eos>":
                    break
                else:
                    decoded_seq.append(list(self.vocab.keys())[list(self.vocab.values()).index(token)])
            decoded.append("".join(decoded_seq))
        return decoded

def analyze_data(rna_seqs: List[str], protein_seqs: List[str], labels: List[str]) -> Tuple[Dict[str, int], Dict[str, int], int]:
    """
    Analyze the data by printing some statistics.

    Args:
        rna_seqs: List of RNA sequences
        protein_seqs: List of protein sequences
        labels: List of binding labels
    """
    print(f"Number of samples: {len(rna_seqs)}")
    rna_lengths = [len(seq) for seq in tqdm(rna_seqs, desc="Analyzing rna length")]
    protein_lengths = [len(seq) for seq in tqdm(protein_seqs, desc="Analyzing protein length")]
    print(f"Max RNA length: {max(rna_lengths)}")
    print(f"Min RNA length: {min(rna_lengths)}")
    print(f"Average RNA length: {np.mean(rna_lengths)}")
    print(f"Max protein length: {max(protein_lengths)}")
    print(f"Min protein length: {min(protein_lengths)}")
    print(f"Average protein length: {np.mean(protein_lengths)}")

    base_count = {
        "A": 0,
        "C": 0,
        "G": 0,
        "U": 0
    }
    for seq in tqdm(rna_seqs, desc="Analyzing RNA base distribution"):
        for base in seq:
            if base in base_count:
                base_count[base] += 1
            else:
                continue
    print(f"RNA base distribution: {base_count}")
    aa_count = {
        "A": 0,
        "C": 0,
        "D": 0,
        "E": 0,
        "F": 0,
        "G": 0,
        "H": 0,
        "I": 0,
        "K": 0,
        "L": 0,
        "M": 0,
        "N": 0,
        "P": 0,
        "Q": 0,
        "R": 0,
        "S": 0,
        "T": 0,
        "V": 0,
        "W": 0,
        "Y": 0
    }
    for seq in tqdm(protein_seqs, desc="Analyzing protein amino acid distribution"):
        for aa in seq:
            if aa in aa_count:
                aa_count[aa] += 1
            else:
                continue
    print(f"Protein amino acid distribution: {aa_count}")

    label_count = 0
    for label in tqdm(labels, desc="Analyzing label distribution"):
        label_count += int(label)
    print(f"Number of positive samples: {label_count}")
    print(f"Number of negative samples: {len(labels) - label_count}")

    return base_count, aa_count, label_count

class RPIDataset(LightningDataModule):
    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 num_workers:int,
                 rna_col: str,
                 protein_col: str,
                 label_col: str,
                 padding: bool,
                 rna_max_length: int,
                 protein_max_length: int,
                 truncation: bool,
                 val_ratio: float=0.1,
                 test_ratio: float=0.1):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.truncation = truncation
        self.raw_data = read_csv(data_path, rna_col, protein_col, label_col)
        self.protein_tokenizer = Tokenizer.protein(padding=padding, max_length=protein_max_length, to_tensor=True)
        self.rna_tokenizer = Tokenizer.rna(padding=padding, max_length=rna_max_length, to_tensor=True)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.embedded_rna = None
        self.rna_mask = None
        self.embedded_protein = None
        self.protein_mask = None
        self.labels = None
        self.embedded_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage=None):
        self._shuffle_data()
        if self.truncation:
            self._truncation()
        else:
            self._select_data()
        self.embedded_rna, self.rna_mask = self.rna_tokenizer.encode(self.raw_data[0])
        self.embedded_protein, self.protein_mask = self.protein_tokenizer.encode(self.raw_data[1])
        self.labels = torch.tensor(self.raw_data[2], dtype=torch.long)
        self.embedded_data = TensorDataset(self.embedded_rna,
                                           self.rna_mask,
                                           self.embedded_protein,
                                           self.protein_mask,
                                           self.labels)
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(self.embedded_data,
                                                                                       [int(len(self.embedded_data))-int(self.val_ratio*len(self.embedded_data))-int(self.test_ratio*len(self.embedded_data)),
                                                                                        int(self.val_ratio*len(self.embedded_data)),
                                                                                        int(self.test_ratio*len(self.embedded_data))])

    def _shuffle_data(self):
        index = np.arange(len(self.raw_data[0]))
        np.random.shuffle(index)
        rna_seqs = [self.raw_data[0][i] for i in tqdm(index, desc="Shuffling RNA sequences")]
        protein_seqs = [self.raw_data[1][i] for i in tqdm(index, desc="Shuffling protein sequences")]
        labels = [self.raw_data[2][i] for i in tqdm(index, desc="Shuffling labels")]
        self.raw_data = (rna_seqs, protein_seqs, labels)

    def _select_data(self):
        original_count = len(self.raw_data[0])
        for i in tqdm(range(len(self.raw_data[0])-1, -1, -1), desc="Selecting data"):
            if len(self.raw_data[0][i]) > self.rna_tokenizer.max_length-2 or len(self.raw_data[1][i]) > self.protein_tokenizer.max_length-2:
                self.raw_data[0].pop(i)
                self.raw_data[1].pop(i)
                self.raw_data[2].pop(i)
        print(f"Selected {len(self.raw_data[0])} samples from {original_count} samples")

    def _truncation(self):
        for i in tqdm(range(len(self.raw_data[0])), desc="Truncating sequences"):
            if len(self.raw_data[0][i]) > self.rna_tokenizer.max_length-2:
                self.raw_data[0][i] = self.raw_data[0][i][:self.rna_tokenizer.max_length-2]
            if len(self.raw_data[1][i]) > self.protein_tokenizer.max_length-2:
                self.raw_data[1][i] = self.raw_data[1][i][:self.protein_tokenizer.max_length-2]
            else:
                continue


    def _create_dataloader(self, data: TensorDataset):
        return DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_data)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_data)

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.test_data)


def write_output(rna_seq: str, protein_seq: str, predict_label: str, path: str=OUTPUT_PATH):
    """
    Write the output to a CSV file.

    Args:
        rna_seq: RNA sequence
        protein_seq: Protein sequence
        predict_label: Predicted label
        path: Path to the output file
    """
    date = pd.Timestamp.now().strftime("%Y%m%d")
    file_path = f'{path}output_{date}.csv'
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("RNA_seq,Protein_seq,Predict_label\n")
    with open(file_path, "a") as f:
        f.write(f"{rna_seq},{protein_seq},{predict_label}\n")
