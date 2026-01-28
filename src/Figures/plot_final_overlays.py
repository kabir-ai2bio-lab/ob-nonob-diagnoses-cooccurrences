"""
Generate publication-quality overlay plots from k-fold training results.

STEP 4: Create 5 specific plots without retraining:
1. AUROC_overlay.png - ROC curves for all 5 models
2. AUPRC_overlay.png - PR curves for all 5 models
3. Loss_overlay.png - Training loss curves
4. ValMetric_overlay.png - Validation AUPRC over epochs
5. ConfusionBars_overlay.png - Confusion matrix metrics

Styling requirements:
- 300 DPI
- Bold labels size 12
- NO titles
- Specific colors per model
"""

import sys
import os
import argparse
import json
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import pandas as pd

# Set matplotlib font properties
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Model colors (EXACT specification)
MODEL_COLORS = {
    'gcn': '#156082',
    'gat': '#E97132',
    'graphsage': '#196B24',
    'hybrid_gcn_gat': '#0F9ED5',
    'hybrid_gcn_graphsage': '#A02B93',
    'graphsage_gat_hybrid': '#E63946'
}

MODEL_NAMES_DISPLAY = {
    'gcn': 'GCN',
    'gat': 'GAT',
    'graphsage': 'GraphSAGE',
    'hybrid_gcn_gat': 'GCN+GAT',
    'hybrid_gcn_graphsage': 'GCN+GraphSAGE',
    'graphsage_gat_hybrid': 'GAT+GraphSAGE'
}

# Model display order
MODEL_ORDER = ['gcn', 'gat', 'graphsage', 'hybrid_gcn_gat', 'hybrid_gcn_graphsage', 'graphsage_gat_hybrid']


class PlotGenerator:
    """Generate all 5 required plots from k-fold results."""
    
    def __init__(self, results_dir: str, out_dir: str = 'figures/new'):
        """
        Args:
            results_dir: Directory containing model subdirectories (e.g., runs/final_kfold/)
            out_dir: Output directory for plots
        """
        self.results_dir = results_dir
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        # Find all model runs
        self.model_runs = self._discover_model_runs()
        
        print(f"\nDiscovered {len(self.model_runs)} model runs:")
        for model_name in self.model_runs.keys():
            print(f"  - {model_name}")
    
    def _discover_model_runs(self) -> Dict[str, str]:
        """
        Discover model run directories.
        
        Returns:
            Dictionary mapping model_name -> run_directory_path
        """
        model_runs = {}
        
        for entry in os.listdir(self.results_dir):
            path = os.path.join(self.results_dir, entry)
            
            if os.path.isdir(path):
                # Check if directory name starts with known model names
                # Sort by length descending to match longer names first (e.g., graphsage_gat_hybrid before graphsage)
                for model_name in sorted(MODEL_COLORS.keys(), key=len, reverse=True):
                    if entry.startswith(model_name):
                        model_runs[model_name] = path
                        break
        
        return model_runs
    
    def generate_all_plots(self):
        """Generate all 5 plots."""
        print(f"\n{'='*80}")
        print("Generating Publication-Quality Plots")
        print(f"{'='*80}")
        
        self.plot_auroc_overlay()
        self.plot_auprc_overlay()
        self.plot_loss_overlay()
        self.plot_val_metric_overlay()
        self.plot_confusion_bars_overlay()
        
        print(f"\nAll plots saved to {self.out_dir}")
    
    def plot_auroc_overlay(self):
        """Plot 1: ROC curves overlay."""
        print("\n[1/5] Generating AUROC_overlay.png...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot in specified order
        for model_name in MODEL_ORDER:
            if model_name not in self.model_runs:
                continue
            run_dir = self.model_runs[model_name]
            # Load predictions from all folds
            all_y_true = []
            all_y_pred = []
            
            fold_dirs = [d for d in os.listdir(run_dir) if d.startswith('fold_')]
            
            for fold_dir in sorted(fold_dirs):
                pred_path = os.path.join(run_dir, fold_dir, 'test_predictions_best.npz')
                
                if os.path.exists(pred_path):
                    data = np.load(pred_path)
                    all_y_true.append(data['y_true'])
                    all_y_pred.append(data['y_pred_proba'])
            
            if all_y_true:
                # Concatenate all folds
                y_true_concat = np.concatenate(all_y_true)
                y_pred_concat = np.concatenate(all_y_pred)
                
                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y_true_concat, y_pred_concat)
                
                # Compute AUROC
                from sklearn.metrics import roc_auc_score
                auroc = roc_auc_score(y_true_concat, y_pred_concat)
                
                # Plot
                ax.plot(
                    fpr, tpr,
                    color=MODEL_COLORS[model_name],
                    linewidth=2.5,
                    label=f"{MODEL_NAMES_DISPLAY[model_name]} (AUROC={auroc:.3f})"
                )
        
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
        
        # Styling
        ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'AUROC_overlay.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ AUROC_overlay.png saved")
    
    def plot_auprc_overlay(self):
        """Plot 2: PR curves overlay."""
        print("\n[2/5] Generating AUPRC_overlay.png...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot in specified order
        for model_name in MODEL_ORDER:
            if model_name not in self.model_runs:
                continue
            run_dir = self.model_runs[model_name]
            # Load predictions from all folds
            all_y_true = []
            all_y_pred = []
            
            fold_dirs = [d for d in os.listdir(run_dir) if d.startswith('fold_')]
            
            for fold_dir in sorted(fold_dirs):
                pred_path = os.path.join(run_dir, fold_dir, 'test_predictions_best.npz')
                
                if os.path.exists(pred_path):
                    data = np.load(pred_path)
                    all_y_true.append(data['y_true'])
                    all_y_pred.append(data['y_pred_proba'])
            
            if all_y_true:
                # Concatenate all folds
                y_true_concat = np.concatenate(all_y_true)
                y_pred_concat = np.concatenate(all_y_pred)
                
                # Compute PR curve (sklearn returns in decreasing recall order)
                precision, recall, _ = precision_recall_curve(y_true_concat, y_pred_concat)
                
                # REVERSE arrays so curve goes from low recall to high recall (left to right)
                # This makes the curve hug upper-right corner like a proper AUPRC plot
                precision = precision[::-1]
                recall = recall[::-1]
                
                # Compute AUPRC
                from sklearn.metrics import average_precision_score
                auprc = average_precision_score(y_true_concat, y_pred_concat)
                
                # Plot
                ax.plot(
                    recall, precision,
                    color=MODEL_COLORS[model_name],
                    linewidth=2.5,
                    label=f"{MODEL_NAMES_DISPLAY[model_name]} (AUPRC={auprc:.3f})"
                )
        
        # Baseline (random classifier)
        baseline = np.mean([np.mean(np.load(os.path.join(run_dir, fold_dir, 'test_predictions_best.npz'))['y_true']) 
                           for run_dir in self.model_runs.values() 
                           for fold_dir in os.listdir(run_dir) if fold_dir.startswith('fold_') 
                           and os.path.exists(os.path.join(run_dir, fold_dir, 'test_predictions_best.npz'))])
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label=f'Baseline ({baseline:.3f})')
        
        # Styling
        ax.set_xlabel('Recall', fontweight='bold', fontsize=12)
        ax.set_ylabel('Precision', fontweight='bold', fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0.5, 1.0])  # Zoom to relevant range where curves are
        ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='black')
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'AUPRC_overlay.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ AUPRC_overlay.png saved")
    
    def plot_loss_overlay(self):
        """Plot 3: Training loss curves overlay."""
        print("\n[3/5] Generating Loss_overlay.png...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot in specified order
        for model_name in MODEL_ORDER:
            if model_name not in self.model_runs:
                continue
            run_dir = self.model_runs[model_name]
            # Load training logs from all folds and average
            all_losses = []
            
            fold_dirs = [d for d in os.listdir(run_dir) if d.startswith('fold_')]
            
            for fold_dir in sorted(fold_dirs):
                log_path = os.path.join(run_dir, fold_dir, 'train_log.csv')
                
                if os.path.exists(log_path):
                    df = pd.read_csv(log_path)
                    all_losses.append(df['loss'].values)
            
            if all_losses:
                # Find minimum length
                min_len = min(len(losses) for losses in all_losses)
                
                # Truncate all to minimum length
                all_losses_truncated = [losses[:min_len] for losses in all_losses]
                
                # Average across folds
                mean_loss = np.mean(all_losses_truncated, axis=0)
                std_loss = np.std(all_losses_truncated, axis=0)
                epochs = np.arange(len(mean_loss))
                
                # Plot mean with shaded std
                ax.plot(
                    epochs, mean_loss,
                    color=MODEL_COLORS[model_name],
                    linewidth=2.5,
                    label=MODEL_NAMES_DISPLAY[model_name]
                )
                ax.fill_between(
                    epochs, mean_loss - std_loss, mean_loss + std_loss,
                    color=MODEL_COLORS[model_name],
                    alpha=0.2
                )
        
        # Styling
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
        ax.set_ylabel('Training Loss', fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'Loss_overlay.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Loss_overlay.png saved")
    
    def plot_val_metric_overlay(self):
        """Plot 4: Validation AUROC over epochs overlay."""
        print("\n[4/5] Generating ValMetric_overlay.png...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot in specified order
        for model_name in MODEL_ORDER:
            if model_name not in self.model_runs:
                continue
            run_dir = self.model_runs[model_name]
            # Load validation AUROC from all folds
            all_val_auroc = []
            
            fold_dirs = [d for d in os.listdir(run_dir) if d.startswith('fold_')]
            
            for fold_dir in sorted(fold_dirs):
                log_path = os.path.join(run_dir, fold_dir, 'train_log.csv')
                
                if os.path.exists(log_path):
                    df = pd.read_csv(log_path)
                    all_val_auroc.append(df['val_auroc'].values)
            
            if all_val_auroc:
                # Find minimum length
                min_len = min(len(auroc) for auroc in all_val_auroc)
                
                # Truncate all to minimum length
                all_val_auroc_truncated = [auroc[:min_len] for auroc in all_val_auroc]
                
                # Average across folds
                mean_auroc = np.mean(all_val_auroc_truncated, axis=0)
                std_auroc = np.std(all_val_auroc_truncated, axis=0)
                epochs = np.arange(len(mean_auroc))
                
                # Plot mean with shaded std
                ax.plot(
                    epochs, mean_auroc,
                    color=MODEL_COLORS[model_name],
                    linewidth=2.5,
                    label=MODEL_NAMES_DISPLAY[model_name]
                )
                ax.fill_between(
                    epochs, mean_auroc - std_auroc, mean_auroc + std_auroc,
                    color=MODEL_COLORS[model_name],
                    alpha=0.2
                )
        
        # Styling
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
        ax.set_ylabel('Validation AUROC', fontweight='bold', fontsize=12)
        ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'ValMetric_overlay.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ ValMetric_overlay.png saved")
    
    def plot_confusion_bars_overlay(self):
        """Plot 5: Confusion matrix TP/TN/FP/FN averaged counts as grouped bar chart."""
        print("\n[5/5] Generating ConfusionBars_overlay.png...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Categories: TP, TN, FP, FN
        categories = ['TP', 'TN', 'FP', 'FN']
        
        # Collect confusion matrix data averaged across all folds
        model_data = {}
        
        for model_name, run_dir in self.model_runs.items():
            # Collect confusion matrix from all folds
            fold_cms = []
            
            fold_dirs = [d for d in os.listdir(run_dir) if d.startswith('fold_')]
            
            for fold_dir in fold_dirs:
                pred_path = os.path.join(run_dir, fold_dir, 'test_predictions_best.npz')
                
                if os.path.exists(pred_path):
                    data = np.load(pred_path)
                    y_true = data['y_true']
                    y_score = data['y_pred_proba']
                    y_pred = (y_score >= 0.5).astype(int)
                    
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    fold_cms.append({'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn})
            
            # Average across folds
            if fold_cms:
                model_data[model_name] = {
                    'TP': np.mean([cm['TP'] for cm in fold_cms]),
                    'TN': np.mean([cm['TN'] for cm in fold_cms]),
                    'FP': np.mean([cm['FP'] for cm in fold_cms]),
                    'FN': np.mean([cm['FN'] for cm in fold_cms])
                }
        
        # Prepare bar positions
        x = np.arange(len(categories))
        width = 0.14
        
        # Plot in specified order
        plot_index = 0
        for model_name in MODEL_ORDER:
            if model_name not in model_data:
                continue
            data = model_data[model_name]
            counts = [data[cat] for cat in categories]
            
            bars = ax.bar(
                x + plot_index * width,
                counts,
                width,
                label=MODEL_NAMES_DISPLAY[model_name],
                color=MODEL_COLORS[model_name],
                edgecolor='black',
                linewidth=1.2
            )
            plot_index += 1
            
            # Add count labels inside bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height / 2,
                    f'{int(count)}',
                    ha='center',
                    va='center',
                    fontweight='bold',
                    fontsize=8,
                    color='white'
                )
        
        # Styling
        ax.set_xlabel('Confusion Matrix Category', fontweight='bold', fontsize=12)
        ax.set_ylabel('Count (Averaged across 5 folds)', fontweight='bold', fontsize=12)
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(categories, fontweight='bold', fontsize=12)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=6, frameon=True, fancybox=False, edgecolor='black')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'ConfusionBars_overlay.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ ConfusionBars_overlay.png saved")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality overlay plots")
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing model run subdirectories')
    parser.add_argument('--out_dir', type=str, default='figures/new',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    plotter = PlotGenerator(
        results_dir=args.results_dir,
        out_dir=args.out_dir
    )
    
    plotter.generate_all_plots()


if __name__ == '__main__':
    main()
