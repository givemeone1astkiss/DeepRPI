DATA_PATH = './data/'
SAVE_PATH = './save/'
LOG_PATH = './logs/'
OUTPUT_PATH = './out/'

AMINO_ACIDS_LONG:list = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]

AMINO_ACIDS: dict = {
    "<UNK>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<PAD>": 3,
    "A": 4,
    "C": 5,
    "D": 6,
    "E": 7,
    "F": 8,
    "G": 9,
    "H": 10,
    "I": 11,
    "K": 12,
    "L": 13,
    "M": 14,
    "N": 15,
    "P": 16,
    "Q": 17,
    "R": 18,
    "S": 19,
    "T": 20,
    "V": 21,
    "W": 22,
    "Y": 23
}

RNA_BASES:dict = {
    "<UNK>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<PAD>": 3,
    "A": 4,
    "C": 5,
    "G": 6,
    "U": 7
}

RNA_BASES_LONG:list = ["ADENINE", "CYTOSINE", "GUANINE", "URACIL"]
