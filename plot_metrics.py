#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot training and validation metrics vs step
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set font
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set plot style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def load_and_process_data():
    """Load and process training and validation data"""
    # Read training data
    train_df = pd.read_csv('csv_logs/training_metrics.csv')
    
    # Read validation data
    val_df = pd.read_csv('csv_logs/validation_metrics.csv')
    
    # Process training data - group by step and calculate mean
    train_processed = train_df.groupby('step').agg({
        'loss': 'mean',
        'accuracy': 'mean', 
        'f1': 'mean',
        'precision': 'mean',
        'recall': 'mean'
    }).reset_index()
    train_processed['type'] = 'Training'
    
    # Process validation data - group by step and calculate mean
    val_processed = val_df.groupby('step').agg({
        'loss': 'mean',
        'accuracy': 'mean',
        'f1': 'mean', 
        'precision': 'mean',
        'recall': 'mean'
    }).reset_index()
    val_processed['type'] = 'Validation'
    
    return train_processed, val_processed

def plot_metrics(train_df, val_df, output_dir):
    """Plot all metrics charts"""
    # Define metrics and their English names
    metrics = {
        'loss': 'Loss',
        'accuracy': 'Accuracy', 
        'f1': 'F1-Score',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Define colors
    train_color = '#2E86AB'  # Blue
    val_color = '#A23B72'    # Red
    
    for i, (metric, title) in enumerate(metrics.items()):
        ax = axes[i]
        
        # Plot training curve
        ax.plot(train_df['step'], train_df[metric], 
                label='Training', color=train_color, linewidth=2, alpha=0.8)
        
        # Plot validation curve
        ax.plot(val_df['step'], val_df[metric], 
                label='Validation', color=val_color, linewidth=2, alpha=0.8)
        
        # Set title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=10)
        
        # Set y-axis range
        if metric == 'loss':
            ax.set_ylim(0, max(train_df[metric].max(), val_df[metric].max()) * 1.1)
        else:
            ax.set_ylim(0, 1)
    
    # Hide the last subplot
    axes[-1].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'training_validation_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_individual_metrics(train_df, val_df, output_dir):
    """Plot individual metrics charts"""
    metrics = {
        'loss': 'Loss',
        'accuracy': 'Accuracy', 
        'f1': 'F1-Score',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    for metric, title in metrics.items():
        plt.figure(figsize=(10, 6))
        
        # Plot training curve
        plt.plot(train_df['step'], train_df[metric], 
                label='Training', color='#2E86AB', linewidth=2, alpha=0.8)
        
        # Plot validation curve
        plt.plot(val_df['step'], val_df[metric], 
                label='Validation', color='#A23B72', linewidth=2, alpha=0.8)
        
        # Set title and labels
        plt.title(f'{title} vs Training Step', fontsize=16, fontweight='bold')
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel(title, fontsize=12)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Set y-axis range
        if metric == 'loss':
            plt.ylim(0, max(train_df[metric].max(), val_df[metric].max()) * 1.1)
        else:
            plt.ylim(0, 1)
        
        # Save plot
        output_path = os.path.join(output_dir, f'{metric}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function"""
    # Create output directory
    output_dir = "metrics_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print("Loading data...")
    train_df, val_df = load_and_process_data()
    
    print("Plotting comprehensive metrics...")
    plot_metrics(train_df, val_df, output_dir)
    
    print("Plotting individual metrics...")
    plot_individual_metrics(train_df, val_df, output_dir)
    
    print("Plotting completed!")
    print(f"Generated files in '{output_dir}' directory:")
    print("- training_validation_metrics.png (comprehensive chart)")
    print("- loss_comparison.png")
    print("- accuracy_comparison.png") 
    print("- f1_comparison.png")
    print("- precision_comparison.png")
    print("- recall_comparison.png")

if __name__ == "__main__":
    main()
