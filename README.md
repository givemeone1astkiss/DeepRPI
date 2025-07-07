# DeepRPI
Deep learning-based RNA-protein interaction prediction tool, official model  for dry lab of iGEM 2025 PekingHSC.

## Project Overview

DeepRPI is a deep learning-based RNA-protein interaction prediction tool. It leverages pretrained protein language models (ESM) and RNA language models (RNABert) to generate sequence embeddings, captures complex relationships between protein and RNA sequences through a bidirectional cross-attention mechanism, and ultimately uses a multilayer perceptron for interaction prediction.

### Key Features

- Uses state-of-the-art pretrained language models for sequence embedding
- Implements bidirectional cross-attention mechanism to capture protein-RNA interaction patterns
- Generates attention heatmaps to visualize key regions of interaction
- Provides complete training, evaluation, and prediction pipeline

## Project Structure

```
DeepRPI/
├── deeprpi/                      # Core modules
│   ├── model/                    # Model-related code
│   │   ├── attention.py          # Cross-attention mechanism implementation
│   │   ├── classifier.py         # Classifier model
│   │   └── embedding.py          # Embedding generation modules
│   ├── utils/                    # Utility functions
│   │   ├── data.py               # Data processing module
│   │   ├── evaluation.py         # Model evaluation module
│   │   ├── lightning_modules.py  # PyTorch Lightning modules
│   │   ├── prediction.py         # Prediction functionality
│   │   └── trainer.py            # Training functionality
│   └── config/                   # Configuration files
├── data/                         # Data directory
├── train.py                      # Training entry script
├── predict.py                    # Prediction entry script
└── evaluate.py                   # Evaluation entry script
```

## Installation

1. Clone the repository:


2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
python train.py --data_path data/your_data.csv --max_epochs 20 --batch_size 16
```

Parameters:
- `--data_path`: Path to training data file
- `--batch_size`: Training batch size
- `--max_epochs`: Maximum number of training epochs
- `--hidden_dim`: Hidden layer dimension
- `--dropout`: Dropout rate

### Predicting Interactions

```bash
python predict.py --protein "MQVDPRLGRTLGKLGKLGRTGRNPQTDKPSLQ" --rna "ACCCGUGUGGUAGCGCAUUAUCGCGCUCACGC"
```

Parameters:
- `--protein`: Protein sequence (amino acid sequence)
- `--rna`: RNA sequence (nucleotide sequence)
- `--checkpoint`: Model checkpoint path (optional, uses pretrained model by default)
- `--output_dir`: Output directory
- `--plot_attention`: Whether to generate attention heatmaps

### Evaluating Model Performance

```bash
python evaluate.py --data_path data/test_data.csv --checkpoint models/best_model.ckpt
```

Parameters:
- `--data_path`: Path to data file
- `--checkpoint`: Model checkpoint path
- `--output_dir`: Output directory
- `--save_attention`: Whether to save attention heatmaps
- `--eval_val`: Whether to evaluate validation set (evaluates test set by default)

## Model Architecture

DeepRPI uses the following components:

1. **Sequence Embedding**:
   - Protein sequence: Generated using ESM-2 pretrained model
   - RNA sequence: Generated using RNABert pretrained model

2. **Cross-Attention Mechanism**: Implements bidirectional protein-RNA attention calculation to capture interaction patterns

3. **Classifier**: Uses a multilayer perceptron for binary classification prediction

## Citation

If you use DeepRPI in your research, please cite our work:

```
@article{DeepRPI2025,
  title={DeepRPI: Deep learning-based RNA-protein interaction prediction with cross-attention mechanism},
  author={PekingHSC-iGEM Team},
  year={2025}
}
```