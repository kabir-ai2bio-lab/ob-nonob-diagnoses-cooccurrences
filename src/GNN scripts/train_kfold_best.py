"""
Final k-fold training with best hyperparameters.

STEP 3: Train all folds with best hyperparameters found from tuning.
- Load pre-generated k-fold splits (NEVER regenerate)
- Train with edge_index_mp per fold (leakage prevention)
- Save: best_checkpoint.pt, test_predictions_best.npz, train_log.csv per fold
"""

import os
import argparse
import csv
from datetime import datetime
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_io import load_graph
from split_manager import load_kfold_splits
from models import GCN, GraphSAGE, GAT, HybridGCNGAT, HybridGCNGraphSAGE, GraphSAGEGATHybrid
from utils import set_seed, get_device, count_parameters
from metrics import compute_metrics_dict
from split_asserts import assert_split_integrity


class KFoldTrainer:
    """K-fold trainer with best hyperparameters."""
    
    def __init__(
        self,
        pkl_path: str,
        splits_path: str,
        model_name: str,
        hyperparams: Dict,
        max_epochs: int = 300,
        patience: int = 40,
        out_dir: str = 'runs/final_kfold',
        seed: int = 42
    ):
        self.pkl_path = pkl_path
        self.splits_path = splits_path
        self.model_name = model_name
        self.hyperparams = hyperparams
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(out_dir, f"{model_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Load graph
        self.edge_index, self.x, self.num_nodes, self.is_obstetric = load_graph(pkl_path)
        
        # Load k-fold splits
        self.folds = load_kfold_splits(splits_path)
        
        self.device = get_device()
        
        # Move to device
        self.x = self.x.to(self.device)
        self.edge_index = self.edge_index.to(self.device)
        
        print(f"\n{'='*80}")
        print(f"K-Fold Training: {model_name}")
        print(f"{'='*80}")
        print(f"Number of folds: {len(self.folds)}")
        print(f"Output directory: {self.run_dir}")
        print(f"Hyperparameters: {hyperparams}")
    
    def train_all_folds(self):
        """Train all folds and save results."""
        all_fold_results = []
        
        for fold_data in self.folds:
            fold_id = fold_data['fold_id']
            
            print(f"\n{'#'*80}", flush=True)
            print(f"Training Fold {fold_id}/{len(self.folds)-1}", flush=True)
            print(f"{'#'*80}", flush=True)
            
            results = self.train_single_fold(fold_data)
            all_fold_results.append(results)
        
        # Save aggregated summary
        self._save_summary(all_fold_results)
        
        print(f"\n{'='*80}", flush=True)
        print("K-Fold Training Complete", flush=True)
        print(f"Results saved to {self.run_dir}", flush=True)
        print(f"{'='*80}", flush=True)
    
    def train_single_fold(self, fold_data: Dict):
        """Train a single fold."""
        fold_id = fold_data['fold_id']
        
        # Create fold directory
        fold_dir = os.path.join(self.run_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Set seed for this fold
        set_seed(self.seed + fold_id)
        
        # Extract data
        train_pos = fold_data['train_pos']
        train_neg = fold_data['train_neg']
        val_pos = fold_data['val_pos']
        val_neg = fold_data['val_neg']
        test_pos = fold_data['test_pos']
        test_neg = fold_data['test_neg']
        edge_index_mp = fold_data['edge_index_mp'].to(self.device)
        
        print(f"Train: {len(train_pos)} pos, {len(train_neg)} neg", flush=True)
        print(f"Val: {len(val_pos)} pos, {len(val_neg)} neg", flush=True)
        print(f"Test: {len(test_pos)} pos, {len(test_neg)} neg", flush=True)
        print(f"\nTraining configuration:", flush=True)
        print(f"  Max epochs allowed: {self.max_epochs}", flush=True)
        print(f"  Early stopping patience: {self.patience}", flush=True)
        print(f"  Early stopping metric: validation AUROC (maximize)", flush=True)
        
        # SAFETY CHECK: Assert split integrity before training
        print("\nRunning split integrity checks...", flush=True)
        assert_split_integrity(
            train_pos, val_pos, test_pos,
            train_neg, val_neg, test_neg,
            edge_index_mp, self.edge_index,
            fold_id=fold_id,
            is_obstetric=self.is_obstetric
        )
        
        # Create model
        model = self._create_model().to(self.device)
        print(f"Model parameters: {count_parameters(model):,}", flush=True)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.hyperparams['lr'],
            weight_decay=self.hyperparams.get('weight_decay', 0.0)
        )
        
        # Prepare edge tensors
        train_pos_t = torch.tensor(train_pos, dtype=torch.long).t().to(self.device)
        train_neg_t = torch.tensor(train_neg, dtype=torch.long).t().to(self.device)
        val_pos_t = torch.tensor(val_pos, dtype=torch.long).t().to(self.device)
        val_neg_t = torch.tensor(val_neg, dtype=torch.long).t().to(self.device)
        test_pos_t = torch.tensor(test_pos, dtype=torch.long).t().to(self.device)
        test_neg_t = torch.tensor(test_neg, dtype=torch.long).t().to(self.device)
        
        train_edge_index = torch.cat([train_pos_t, train_neg_t], dim=1)
        train_labels = torch.cat([
            torch.ones(train_pos_t.shape[1]),
            torch.zeros(train_neg_t.shape[1])
        ]).to(self.device)
        
        val_edge_index = torch.cat([val_pos_t, val_neg_t], dim=1)
        val_labels = torch.cat([
            torch.ones(val_pos_t.shape[1]),
            torch.zeros(val_neg_t.shape[1])
        ]).to(self.device)
        
        test_edge_index = torch.cat([test_pos_t, test_neg_t], dim=1)
        test_labels = torch.cat([
            torch.ones(test_pos_t.shape[1]),
            torch.zeros(test_neg_t.shape[1])
        ]).to(self.device)
        
        # Training loop
        best_val_auroc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        
        train_log = []
        
        for epoch in range(self.max_epochs):
            # ASSERTION: Epoch must be within bounds (defensive check)
            assert epoch < self.max_epochs, (
                f"FATAL: Epoch {epoch} >= max_epochs {self.max_epochs}. Loop invariant violated."
            )
            
            # Train
            model.train()
            optimizer.zero_grad()
            
            z = model(self.x, edge_index_mp)
            train_scores = model.decode(z, train_edge_index)
            loss = F.binary_cross_entropy_with_logits(train_scores, train_labels)
            
            loss.backward()
            optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                z = model(self.x, edge_index_mp)
                
                train_scores = model.decode(z, train_edge_index)
                train_probs = torch.sigmoid(train_scores).cpu().numpy()
                train_metrics = compute_metrics_dict(train_labels.cpu().numpy(), train_probs, prefix='train')
                
                val_scores = model.decode(z, val_edge_index)
                val_probs = torch.sigmoid(val_scores).cpu().numpy()
                val_metrics = compute_metrics_dict(val_labels.cpu().numpy(), val_probs, prefix='val')
                
                test_scores = model.decode(z, test_edge_index)
                test_probs = torch.sigmoid(test_scores).cpu().numpy()
                test_metrics = compute_metrics_dict(test_labels.cpu().numpy(), test_probs, prefix='test')
            
            # Log
            log_entry = {
                'epoch': epoch,
                'loss': loss.item(),
                **train_metrics,
                **val_metrics,
                **test_metrics
            }
            train_log.append(log_entry)
            
            # Check improvement
            val_auroc = val_metrics['val_auroc']
            
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_epoch = epoch
                epochs_no_improve = 0
                
                # ASSERTION: Best epoch must be within bounds
                assert best_epoch < self.max_epochs, (
                    f"FATAL: Attempting to save checkpoint for epoch {best_epoch} >= max_epochs {self.max_epochs}"
                )
                
                # Save best checkpoint
                checkpoint_path = os.path.join(fold_dir, 'best_checkpoint.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hyperparams': self.hyperparams,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics
                }, checkpoint_path)
                
                # Save best validation predictions
                val_pred_path = os.path.join(fold_dir, 'val_predictions_best.npz')
                np.savez(
                    val_pred_path,
                    y_true=val_labels.cpu().numpy(),
                    y_pred_proba=val_probs,
                    val_pos=val_pos,
                    val_neg=val_neg
                )
                
                # Save best test predictions
                pred_path = os.path.join(fold_dir, 'test_predictions_best.npz')
                np.savez(
                    pred_path,
                    y_true=test_labels.cpu().numpy(),
                    y_pred_proba=test_probs,
                    test_pos=test_pos,
                    test_neg=test_neg
                )
            else:
                epochs_no_improve += 1
            
            # Print progress
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                      f"Val AUROC: {val_auroc:.4f} | "
                      f"Test AUROC: {test_metrics['test_auroc']:.4f} | "
                      f"Test AUPRC: {test_metrics['test_auprc']:.4f}", flush=True)
            
            # Early stopping
            if epochs_no_improve >= self.patience:
                stopped_epoch = epoch
                print(f"\nEarly stopping at epoch {stopped_epoch} (best epoch: {best_epoch})", flush=True)
                break
        else:
            # Loop completed without early stopping
            stopped_epoch = self.max_epochs - 1
        
        # CRITICAL ASSERTION: Verify epoch bounds were respected
        assert best_epoch < self.max_epochs, (
            f"FATAL: best_epoch ({best_epoch}) >= max_epochs ({self.max_epochs}). "
            f"This should NEVER happen. Training loop violated epoch cap."
        )
        assert stopped_epoch < self.max_epochs, (
            f"FATAL: stopped_epoch ({stopped_epoch}) >= max_epochs ({self.max_epochs}). "
            f"This should NEVER happen. Training loop violated epoch cap."
        )
        
        print(f"\n[FOLD {fold_id}] Training stopped at epoch: {stopped_epoch}", flush=True)
        print(f"[FOLD {fold_id}] Best epoch selected: {best_epoch}", flush=True)
        print(f"[FOLD {fold_id}] Epochs within bounds: VERIFIED ({best_epoch} < {self.max_epochs})", flush=True)
        
        # Save training log
        log_path = os.path.join(fold_dir, 'train_log.csv')
        with open(log_path, 'w', newline='') as f:
            if train_log:
                writer = csv.DictWriter(f, fieldnames=train_log[0].keys())
                writer.writeheader()
                writer.writerows(train_log)
        
        # Load best metrics
        best_metrics = train_log[best_epoch]
        
        print(f"\nBest epoch: {best_epoch}", flush=True)
        print(f"Best val AUROC: {best_metrics['val_auroc']:.4f}", flush=True)
        print(f"Test AUROC: {best_metrics['test_auroc']:.4f}", flush=True)
        print(f"Test AUPRC: {best_metrics['test_auprc']:.4f}", flush=True)
        
        return {
            'fold_id': fold_id,
            'best_epoch': best_epoch,
            'best_val_auroc': best_metrics['val_auroc'],
            'test_metrics': {
                'auroc': best_metrics['test_auroc'],
                'auprc': best_metrics['test_auprc'],
                'accuracy': best_metrics['test_accuracy'],
                'f1': best_metrics['test_f1'],
                'mcc': best_metrics['test_mcc']
            }
        }
    
    def _create_model(self):
        """Create model instance."""
        in_channels = self.x.shape[1]
        
        if self.model_name == 'gcn':
            return GCN(
                in_channels=in_channels,
                hidden_channels=self.hyperparams['hidden_channels'],
                out_channels=self.hyperparams['out_channels'],
                num_layers=self.hyperparams['num_layers'],
                dropout=self.hyperparams['dropout']
            )
        elif self.model_name == 'graphsage':
            return GraphSAGE(
                in_channels=in_channels,
                hidden_channels=self.hyperparams['hidden_channels'],
                out_channels=self.hyperparams['out_channels'],
                num_layers=self.hyperparams['num_layers'],
                dropout=self.hyperparams['dropout'],
                aggr=self.hyperparams.get('aggr', 'mean')
            )
        elif self.model_name == 'gat':
            return GAT(
                in_channels=in_channels,
                hidden_channels=self.hyperparams['hidden_channels'],
                out_channels=self.hyperparams['out_channels'],
                num_layers=self.hyperparams['num_layers'],
                num_heads=self.hyperparams.get('num_heads', 4),
                dropout=self.hyperparams['dropout']
            )
        elif self.model_name == 'hybrid_gcn_gat':
            return HybridGCNGAT(
                in_channels=in_channels,
                hidden_channels=self.hyperparams['hidden_channels'],
                out_channels=self.hyperparams['out_channels'],
                num_layers=self.hyperparams['num_layers'],
                gcn_dropout=self.hyperparams.get('gcn_dropout', self.hyperparams['dropout']),
                gat_dropout=self.hyperparams.get('gat_dropout', self.hyperparams['dropout']),
                gat_heads=self.hyperparams.get('gat_heads', 4),
                fusion_method=self.hyperparams.get('fusion_method', 'projected_concat')
            )
        elif self.model_name == 'hybrid_gcn_graphsage':
            return HybridGCNGraphSAGE(
                in_channels=in_channels,
                hidden_channels=self.hyperparams['hidden_channels'],
                out_channels=self.hyperparams['out_channels'],
                num_layers=self.hyperparams['num_layers'],
                gcn_dropout=self.hyperparams.get('gcn_dropout', self.hyperparams['dropout']),
                sage_dropout=self.hyperparams.get('sage_dropout', self.hyperparams['dropout']),
                sage_aggr=self.hyperparams.get('sage_aggr', 'mean'),
                fusion_method=self.hyperparams.get('fusion_method', 'projected_concat')
            )
        elif self.model_name == 'graphsage_gat_hybrid':
            return GraphSAGEGATHybrid(
                in_channels=in_channels,
                sage_hidden_channels=self.hyperparams.get('sage_hidden_channels', 128),
                sage_num_layers=self.hyperparams.get('sage_num_layers', 2),
                sage_aggr=self.hyperparams.get('sage_aggr', 'mean'),
                dropout=self.hyperparams.get('sage_dropout', self.hyperparams.get('dropout', 0.3)),
                gat_num_layers=self.hyperparams.get('gat_num_layers', 1),
                gat_hidden_channels=self.hyperparams.get('gat_hidden_channels', 64),
                gat_heads=self.hyperparams.get('gat_num_heads', 4),
                gat_dropout=self.hyperparams.get('gat_dropout', self.hyperparams.get('dropout', 0.3)),
                out_channels=self.hyperparams['out_channels']
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _save_summary(self, all_fold_results):
        """Save aggregated summary across folds."""
        import json
        
        # Concatenate predictions across all folds for overlay plotting
        all_test_y_true = []
        all_test_y_pred = []
        all_val_y_true = []
        all_val_y_pred = []
        
        for fold_idx in range(len(self.folds)):
            fold_dir = os.path.join(self.run_dir, f"fold_{fold_idx}")
            
            # Load test predictions
            test_preds = np.load(os.path.join(fold_dir, 'test_predictions_best.npz'))
            all_test_y_true.append(test_preds['y_true'])
            all_test_y_pred.append(test_preds['y_pred_proba'])
            
            # Load validation predictions
            val_preds = np.load(os.path.join(fold_dir, 'val_predictions_best.npz'))
            all_val_y_true.append(val_preds['y_true'])
            all_val_y_pred.append(val_preds['y_pred_proba'])
        
        # Save concatenated predictions for plotting WITHOUT retraining
        concat_preds_path = os.path.join(self.run_dir, 'test_predictions_all_folds.npz')
        np.savez(
            concat_preds_path,
            y_true=np.concatenate(all_test_y_true),
            y_pred_proba=np.concatenate(all_test_y_pred)
        )
        
        concat_val_path = os.path.join(self.run_dir, 'val_predictions_all_folds.npz')
        np.savez(
            concat_val_path,
            y_true=np.concatenate(all_val_y_true),
            y_pred_proba=np.concatenate(all_val_y_pred)
        )
        
        print(f"\n✓ Concatenated predictions saved:", flush=True)
        print(f"  {concat_preds_path}", flush=True)
        print(f"  {concat_val_path}", flush=True)
        
        # Extract test metrics
        test_auroc = [r['test_metrics']['auroc'] for r in all_fold_results]
        test_auprc = [r['test_metrics']['auprc'] for r in all_fold_results]
        test_acc = [r['test_metrics']['accuracy'] for r in all_fold_results]
        test_f1 = [r['test_metrics']['f1'] for r in all_fold_results]
        test_mcc = [r['test_metrics']['mcc'] for r in all_fold_results]
        
        summary = {
            'model_name': self.model_name,
            'hyperparams': self.hyperparams,
            'num_folds': len(self.folds),
            'seed': self.seed,
            'fold_results': all_fold_results,
            'aggregated_test_metrics': {
                'auroc_mean': float(np.mean(test_auroc)),
                'auroc_std': float(np.std(test_auroc)),
                'auprc_mean': float(np.mean(test_auprc)),
                'auprc_std': float(np.std(test_auprc)),
                'accuracy_mean': float(np.mean(test_acc)),
                'accuracy_std': float(np.std(test_acc)),
                'f1_mean': float(np.mean(test_f1)),
                'f1_std': float(np.std(test_f1)),
                'mcc_mean': float(np.mean(test_mcc)),
                'mcc_std': float(np.std(test_mcc))
            }
        }
        
        summary_path = os.path.join(self.run_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}", flush=True)
        print("AGGREGATED TEST METRICS (across folds)", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"AUROC: {summary['aggregated_test_metrics']['auroc_mean']:.4f} ± {summary['aggregated_test_metrics']['auroc_std']:.4f}", flush=True)
        print(f"AUPRC: {summary['aggregated_test_metrics']['auprc_mean']:.4f} ± {summary['aggregated_test_metrics']['auprc_std']:.4f}", flush=True)
        print(f"Accuracy: {summary['aggregated_test_metrics']['accuracy_mean']:.4f} ± {summary['aggregated_test_metrics']['accuracy_std']:.4f}", flush=True)
        print(f"F1: {summary['aggregated_test_metrics']['f1_mean']:.4f} ± {summary['aggregated_test_metrics']['f1_std']:.4f}", flush=True)
        print(f"MCC: {summary['aggregated_test_metrics']['mcc_mean']:.4f} ± {summary['aggregated_test_metrics']['mcc_std']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="K-fold training with best hyperparameters")
    parser.add_argument('--pkl', type=str, required=True, help='Path to graph pickle')
    parser.add_argument('--splits', type=str, required=True, help='Path to k-fold splits .npz file')
    parser.add_argument('--model', type=str, required=True,
                        choices=['gcn', 'graphsage', 'gat', 'hybrid_gcn_gat', 'hybrid_gcn_graphsage'],
                        help='Model name')
    parser.add_argument('--hyperparams_file', type=str, required=True,
                        help='Path to JSON file with best hyperparameters')
    parser.add_argument('--max_epochs', type=int, default=300, help='Max training epochs')
    parser.add_argument('--patience', type=int, default=40, help='Early stopping patience')
    parser.add_argument('--out_dir', type=str, default='runs/final_kfold', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load hyperparameters
    import json
    with open(args.hyperparams_file, 'r') as f:
        hyperparams = json.load(f)
    
    # Train
    trainer = KFoldTrainer(
        pkl_path=args.pkl,
        splits_path=args.splits,
        model_name=args.model,
        hyperparams=hyperparams,
        max_epochs=args.max_epochs,
        patience=args.patience,
        out_dir=args.out_dir,
        seed=args.seed
    )
    
    trainer.train_all_folds()


if __name__ == '__main__':
    main()
