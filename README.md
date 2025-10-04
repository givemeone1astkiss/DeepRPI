# DeepRPI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Deep learning-based RNA-protein interaction prediction tool** - Official model for dry lab of iGEM 2025 PekingHSC.

## Project Overview

DeepRPI is a deep learning tool for predicting RNA-protein interactions. It leverages cutting-edge pretrained language models (ESM-2 for proteins and RNABert for RNA) to generate high-quality sequence embeddings, captures complex relationships between protein and RNA sequences through a sophisticated bidirectional cross-attention mechanism, and uses a multilayer perceptron for accurate interaction prediction.

### Key Features

- 🧬 **Advanced Pretrained Models**: Uses ESM-2 for protein sequence embedding and RNABert for RNA sequence embedding
- 🔗 **Bidirectional Cross-Attention**: Implements sophisticated attention mechanism to capture protein-RNA interaction patterns
- 📊 **Visualization**: Generates attention heatmaps to visualize key regions of interaction
- 🚀 **Complete Pipeline**: Provides comprehensive training, evaluation, and prediction pipeline
- 🎯 **High Accuracy**: Achieves state-of-the-art performance on benchmark datasets
- ⚡ **Easy to Use**: Simple command-line interface with extensive documentation

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
│       ├── glob.py               # Global configuration
│       └── seeds.py              # Random seed configuration
├── data/                         # Data directory
│   └── NPInter2.csv             # NPInter2 dataset
├── train.py                      # Training entry script
├── predict.py                    # Prediction entry script
├── evaluate.py                   # Evaluation entry script
├── plot_metrics.py              # Training metrics visualization script
├── run.sh                        # Automated training script
├── model_checkpoint.ckpt         # Pretrained model checkpoint
└── requirements.txt              # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Quick Install

1. **Clone the repository:**
```bash
git clone https://github.com/givemeone1astkiss/DeepRPI.git
cd DeepRPI
```

2. **Create virtual environment (recommended):**
```bash
python -m venv deeprpi_env
source deeprpi_env/bin/activate  # Linux/Mac
# or
deeprpi_env\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import fair_esm; print('ESM installed successfully')"
```

### Dependencies

- `torch>=2.0.0` - Deep learning framework
- `fair-esm==2.0.0` - Protein language model
- `multimolecule>=0.0.6` - RNA language model
- `pytorch-lightning>=2.0.0` - Training framework
- `scikit-learn>=1.0.0` - Machine learning utilities
- `matplotlib>=3.5.0` - Visualization
- `seaborn>=0.12.0` - Statistical visualization
- `numpy>=1.26.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `tqdm>=4.60.0` - Progress bars
- `tensorboard>=2.10.0` - Training visualization

## Quick Start

### 1. Training a Model

```bash
# Basic training
python train.py --data_path data/NPInter2.csv --max_epochs 20 --batch_size 16

# Advanced training with custom parameters
python train.py \
    --data_path data/your_data.csv \
    --batch_size 8 \
    --max_epochs 50 \
    --hidden_dim 512 \
    --dropout 0.2 \
    --model_seed 123
```

**Key Parameters:**
- `--data_path`: Path to training data file (CSV format)
- `--batch_size`: Training batch size (default: 8)
- `--max_epochs`: Maximum number of training epochs (default: 10)
- `--hidden_dim`: Hidden layer dimension (default: 256)
- `--dropout`: Dropout rate (default: 0.1)
- `--model_seed`: Random seed for model initialization (default: 42)
- `--data_split_seed`: Random seed for data splitting (default: 42)
- `--num_workers`: Number of data loading threads (default: 4)

### 2. Predicting Interactions

```bash
# Basic prediction
python predict.py \
    --protein "MQVDPRLGRTLGKLGKLGRTGRNPQTDKPSLQ" \
    --rna "ACCCGUGUGGUAGCGCAUUAUCGCGCUCACGC"

# Prediction with custom model and attention visualization
python predict.py \
    --protein "MQVDPRLGRTLGKLGKLGRTGRNPQTDKPSLQ" \
    --rna "ACCCGUGUGGUAGCGCAUUAUCGCGCUCACGC" \
    --checkpoint models/my_model.ckpt \
    --output_dir my_predictions \
    --plot_attention
```

**Key Parameters:**
- `--protein`: Protein sequence (amino acid sequence) - **Required**
- `--rna`: RNA sequence (nucleotide sequence) - **Required**
- `--checkpoint`: Model checkpoint path (default: model_checkpoint.ckpt)
- `--output_dir`: Output directory (default: prediction_results)
- `--plot_attention`: Generate attention heatmaps (default: True)

### 3. Evaluating Model Performance

```bash
# Evaluate test set
python evaluate.py --data_path data/test_data.csv --checkpoint models/best_model.ckpt

# Evaluate validation set with attention visualization
python evaluate.py \
    --data_path data/validation_data.csv \
    --checkpoint models/best_model.ckpt \
    --eval_val \
    --output_dir val_results \
    --save_attention
```

**Key Parameters:**
- `--data_path`: Path to data file - **Required**
- `--checkpoint`: Model checkpoint path (optional)
- `--output_dir`: Output directory (default: evaluation_results)
- `--save_attention`: Save attention heatmaps (default: True)
- `--eval_val`: Evaluate validation set instead of test set (default: False)
- `--data_split_seed`: Random seed for data splitting (must match training, default: 42)

### 4. Automated Training

Use the provided script for automated training:

```bash
# Make script executable
chmod +x run.sh

# Run automated training
./run.sh
```

This will start training in the background with optimized parameters and save logs to `logs/` directory. The script uses GPU 1 and includes the following default parameters:
- Batch size: 2
- Max epochs: 10
- Hidden dimension: 256
- Dropout: 0.1
- Seeds: 42 (both model and data split)

### 5. Visualization

Plot training metrics and curves:

```bash
# Plot comprehensive training metrics
python plot_metrics.py

# The script will generate:
# - training_validation_metrics.png (comprehensive chart)
# - Individual metric comparison charts (loss, accuracy, f1, precision, recall)
```

## Model Architecture

DeepRPI employs a sophisticated deep learning architecture with the following key components:

### 1. Sequence Embedding
- **Protein Sequences**: Processed using ESM-2 (Evolutionary Scale Modeling) pretrained model
  - Generates 1280-dimensional embeddings
  - Captures evolutionary and structural information
- **RNA Sequences**: Processed using RNABert pretrained model
  - Generates 120-dimensional embeddings
  - Captures RNA secondary structure and sequence patterns

### 2. Cross-Attention Mechanism
- **Bidirectional Attention**: Computes attention weights between protein and RNA sequences
- **Multi-Head Attention**: Uses 8 attention heads for comprehensive pattern capture
- **Residual Connections**: Ensures stable gradient flow during training
- **Layer Normalization**: Improves training stability and convergence

### 3. Classification Head
- **Dimension Alignment**: Aligns RNA embeddings (120D) to protein embedding dimension (1280D)
- **Attention Pooling**: Uses learnable attention mechanisms to aggregate sequence information
- **Multilayer Perceptron**: Processes fused features for binary classification
- **Dropout Regularization**: Prevents overfitting during training

### 4. Training Strategy
- **Optimizer**: AdamW with learning rate 1e-4
- **Loss Function**: Binary cross-entropy loss
- **Early Stopping**: Prevents overfitting based on validation performance
- **Gradient Clipping**: Ensures training stability

## Data Format

DeepRPI expects CSV files with the following columns:

```csv
rna_sequence,protein_sequence,label
ACCCGUGUGGUAGCGCAUUAUCGCGCUCACGC,MQVDPRLGRTLGKLGKLGRTGRNPQTDKPSLQ,1
AUGCGCAUUAUCGCGCUCACGC,MQVDPRLGRTLGKLGKLGRTGRNPQTDKPSLQ,0
```

- `rna_sequence`: RNA nucleotide sequence (A, U, G, C)
- `protein_sequence`: Protein amino acid sequence (standard 20 amino acids)
- `label`: Binary label (0 = no interaction, 1 = interaction)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use DeepRPI in your research, please cite our work:

```bibtex
@article{DeepRPI2025,
  title={DeepRPI: Deep learning-based RNA-protein interaction prediction},
  author={PekingHSC-iGEM Team},
  year={2025}
}
```

## Acknowledgments

- **ESM-2**: Facebook AI Research for protein language models
- **RNABert**: Hugging Face for RNA language models
- **PyTorch Lightning**: Lightning AI for training framework
- **iGEM 2025**: PekingHSC team for project development

## Contact

- **Email**: 2310307322@stu.pku.edu.cn

---
