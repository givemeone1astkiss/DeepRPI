DATA_PATH = './data/'
SAVE_PATH = './save/'
LOG_PATH = './logs/'
OUTPUT_PATH = './out/'

AMINO_ACIDS_LONG:list = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]

AMINO_ACIDS: dict = {
    "<bos>": 0,
    "<pad>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "L": 4,
    "A": 5,
    "G": 6,
    "V": 7,
    "S": 8,
    "E": 9,
    "R": 10,
    "T": 11,
    "I": 12,
    "D": 13,
    "P": 14,
    "K": 15,
    "Q": 16,
    "N": 17,
    "F": 18,
    "Y": 19,
    "M": 20,
    "H": 21,
    "W": 22,
    "C": 23,
}

RNA_BASES:dict = {
    "<bos>": 0,
    "<pad>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "A": 4,
    "C": 5,
    "G": 6,
    "U": 7
}

RNA_BASES_LONG:list = ["ADENINE", "CYTOSINE", "GUANINE", "URACIL"]
