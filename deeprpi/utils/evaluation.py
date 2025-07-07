import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, average_precision_score, auc,
    confusion_matrix, classification_report
)

from deeprpi.utils.lightning_modules import ProteinRNALightningModule
from deeprpi.utils.data import RPIDataset

def set_seed(seed=42):
    """
    Set all random seeds to ensure reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_attention(attention_weights, title, save_path):
    """
    Plot and save attention heatmap.
    
    Args:
        attention_weights: Attention weight matrix
        title: Chart title
        save_path: Save path
    """
    if attention_weights is None:
        print(f"Warning: No attention weights available for {title}")
        return
    
    # Get attention matrix shape
    attention_cpu = attention_weights.cpu().detach().numpy()
    shape = attention_cpu.shape
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    
    # Generate more meaningful axis labels
    if len(shape) == 2:
        rows, cols = shape
        # Create tick marks, select subset if dimensions are large
        if rows > 10:
            row_ticks = np.linspace(0, rows-1, min(10, rows)).astype(int)
            row_labels = [str(i) for i in row_ticks]
        else:
            row_ticks = np.arange(rows)
            row_labels = [str(i) for i in range(rows)]
            
        if cols > 10:
            col_ticks = np.linspace(0, cols-1, min(10, cols)).astype(int)
            col_labels = [str(i) for i in col_ticks]
        else:
            col_ticks = np.arange(cols)
            col_labels = [str(i) for i in range(cols)]
        
        # Draw heatmap and set ticks
        heatmap = sns.heatmap(
            attention_cpu, 
            cmap='viridis',
            annot=rows*cols <= 100,  # Show values when not too many cells
            fmt='.2f', 
            cbar=True
        )
        
        plt.xticks(col_ticks, col_labels)
        plt.yticks(row_ticks, row_labels)
        plt.xlabel('Target position')
        plt.ylabel('Source position')
    else:
        # For non-2D cases, simply show the first plane
        heatmap = sns.heatmap(
            attention_cpu.reshape(shape[0], -1), 
            cmap='viridis',
            cbar=True
        )
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_dataset(data_file, output_dir, is_val=False, save_attention=True, checkpoint_path=None):
    """
    Evaluate a dataset and generate a results report.
    
    Args:
        data_file: Path to the data file
        output_dir: Output directory
        is_val: Whether to use validation set (otherwise test set)
        save_attention: Whether to save attention plots
        checkpoint_path: Model checkpoint path
        
    Returns:
        dict: Performance metrics
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Set up output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up output file paths
    set_type = "validation" if is_val else "test"
    output_file = output_dir / f"{set_type}_results.csv"
    metrics_file = output_dir / f"{set_type}_metrics.txt"
    roc_curve_file = output_dir / f"{set_type}_roc_curve.png"
    pr_curve_file = output_dir / f"{set_type}_pr_curve.png"
    confusion_matrix_file = output_dir / f"{set_type}_confusion_matrix.png"
    
    print(f"Loading dataset: {data_file}")
    
    # Create dataset
    dataset = RPIDataset(
        data_path=data_file,
        batch_size=8,
        num_workers=4,
        rna_col='RNA_aa_code',
        protein_col='target_aa_code',
        label_col='Y',
        padding=True,
        rna_max_length=220,
        protein_max_length=500,
        truncation=True,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Setup dataset
    dataset.setup()
    
    # Get dataloader
    if is_val:
        dataloader = dataset.val_dataloader()
        print("Using validation set for evaluation")
    else:
        dataloader = dataset.test_dataloader()
        print("Using test set for evaluation")
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ProteinRNALightningModule(
        output_dim=1280,
        hidden_dim=256,
        dropout=0.1
    )
    
    # Load checkpoint
    if checkpoint_path is None:
        # Try to find a checkpoint in the default location
        checkpoint_path = 'lightning_logs/protein_rna_classifier/version_0/checkpoints/best_model-epoch=00-val_f1=0.8257.ckpt'
    
    print(f"Using model checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    try:
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise RuntimeError("Model loading failed, check structure differences")
    
    # Set to evaluation mode
    model.eval()
    model.to(device)
    
    # Create attention map save directory
    attention_dir = Path("attention_maps")
    attention_dir.mkdir(exist_ok=True, parents=True)
    
    # Store results
    results = []
    all_labels = []
    all_probs = []
    
    # Start evaluation
    print("Starting evaluation...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx+1}/{len(dataloader)}...")
        
        # Process batch
        rna_seqs, protein_seqs, labels = model._process_batch(batch)
        
        # Predict
        with torch.no_grad():
            logits, protein_attention, rna_attention = model(protein_seqs, rna_seqs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            
            # Save attention heatmaps for first 5 batches
            if save_attention and batch_idx < 5 and protein_attention is not None and rna_attention is not None:
                try:
                    plot_attention(
                        protein_attention[0],  # Take attention from first sample
                        f"Protein to RNA Attention (Batch {batch_idx})",
                        attention_dir / f"{set_type}_protein_attention_batch_{batch_idx}.png"
                    )
                    plot_attention(
                        rna_attention[0],  # Take attention from first sample
                        f"RNA to Protein Attention (Batch {batch_idx})",
                        attention_dir / f"{set_type}_rna_attention_batch_{batch_idx}.png"
                    )
                except Exception as e:
                    print(f"Could not plot attention: {e}")
        
        # Collect evaluation data
        labels_cpu = labels.cpu().numpy()
        probs_cpu = probs.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        
        all_labels.extend(labels_cpu)
        all_probs.extend(probs_cpu)
        
        # Add results to list
        for i in range(len(labels)):
            results.append({
                'label': int(labels_cpu[i]),
                'probability': float(probs_cpu[i]),
                'prediction': int(preds_cpu[i])
            })
    
    # Convert to numpy arrays for metric calculation
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Predictions based on threshold 0.5
    predictions = (all_probs >= 0.5).astype(int)
    
    # Calculate performance metrics
    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Calculate PR curve and average precision
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    average_precision = average_precision_score(all_labels, all_probs)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{set_type.capitalize()} Set ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_curve_file, dpi=300)
    plt.close()
    
    # Plot PR curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AP = {average_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{set_type.capitalize()} Set PR Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_curve_file, dpi=300)
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['No Interaction', 'Interaction'],
        yticklabels=['No Interaction', 'Interaction']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{set_type.capitalize()} Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig(confusion_matrix_file, dpi=300)
    plt.close()
    
    # Save performance metrics to file
    with open(metrics_file, 'w') as f:
        f.write(f"{set_type.capitalize()} Set Evaluation Results\n")
        f.write(f"Dataset path: {data_file}\n")
        f.write(f"Model checkpoint: {checkpoint_path}\n")
        f.write(f"Sample count: {len(all_labels)}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"PR Average Precision (AP): {average_precision:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"True Positive (TP): {tp}\n")
        f.write(f"False Positive (FP): {fp}\n")
        f.write(f"True Negative (TN): {tn}\n")
        f.write(f"False Negative (FN): {fn}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, predictions))
    
    # Save detailed results to CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print(f"Performance metrics saved to: {metrics_file}")
    
    # Return main performance metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'auc': roc_auc,
        'ap': average_precision
    } 