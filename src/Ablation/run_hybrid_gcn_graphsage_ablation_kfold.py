#!/usr/bin/env python3
"""
K-Fold Ablation Study for Hybrid GCN+GraphSAGE Model

Runs ablations across all k folds and reports mean ± std for each metric.
"""

import os
import sys
import json
import argparse
import csv
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_io import load_graph
from split_manager import load_kfold_splits
from split_asserts import assert_split_integrity
from models import HybridGCNGraphSAGE
from utils import set_seed, get_device, count_parameters
from metrics import compute_metrics_dict


def get_top_predictions(
    edge_index: torch.Tensor,
    probabilities: np.ndarray,
    labels: np.ndarray,
    node_names: list = None,
    top_k: int = 20
) -> Dict:
    """
    Get top K predictions for positive and negative classes.
    
    Args:
        edge_index: Edge pairs (2, num_edges)
        probabilities: Predicted probabilities
        labels: True labels
        node_names: List of node names/IDs (optional)
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with top predictions
    """
    # Top K positive predictions (highest probabilities)
    pos_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_positive = []
    for i in pos_indices:
        src_idx = int(edge_index[0, i])
        tgt_idx = int(edge_index[1, i])
        pred = {
            "source": src_idx,
            "target": tgt_idx,
            "prob": float(probabilities[i]),
            "true_label": int(labels[i])
        }
        if node_names is not None:
            pred["source_name"] = node_names[src_idx]
            pred["target_name"] = node_names[tgt_idx]
        top_positive.append(pred)
    
    # Top K negative predictions (lowest probabilities)
    neg_indices = np.argsort(probabilities)[:top_k]
    top_negative = []
    for i in neg_indices:
        src_idx = int(edge_index[0, i])
        tgt_idx = int(edge_index[1, i])
        pred = {
            "source": src_idx,
            "target": tgt_idx,
            "prob": float(probabilities[i]),
            "true_label": int(labels[i])
        }
        if node_names is not None:
            pred["source_name"] = node_names[src_idx]
            pred["target_name"] = node_names[tgt_idx]
        top_negative.append(pred)
    
    return {
        "top_20_positive": top_positive,
        "top_20_negative": top_negative
    }


def load_concept_names(concept_csv_path: str, node_ids: List[str]) -> Dict[str, str]:
    """Load concept names from concept.csv for given node IDs."""
    try:
        df = pd.read_csv(concept_csv_path, dtype={'concept_id': str}, low_memory=False)
        df_filtered = df[df['concept_id'].isin(node_ids)]
        concept_map = dict(zip(df_filtered['concept_id'], df_filtered['concept_name']))
        print(f"  Loaded {len(concept_map)} concept names from {concept_csv_path}")
        return concept_map
    except Exception as e:
        print(f"  Warning: Could not load concept names: {e}")
        return {}


def apply_feature_ablation(X: torch.Tensor, is_obstetric: np.ndarray, ablation_type: str) -> torch.Tensor:
    """Apply feature ablation to the feature matrix."""
    print(f"\nApplying feature ablation: {ablation_type}")
    
    if ablation_type == "none":
        X_ablated = X.clone()
        print(f"  Using all {X.shape[1]} features")
        
    elif ablation_type == "no_community_flag":
        X_ablated = X.clone()
        X_ablated[:, 5] = 0
        print(f"  Zeroed out 'in_obstetric_community' feature (index 5)")
        
    elif ablation_type == "local_only":
        indices = [0, 6, 1]
        X_ablated = X[:, indices]
        print(f"  Using only local features: degree, log_degree, clustering_coeff ({X_ablated.shape[1]} features)")
        
    elif ablation_type == "global_only":
        indices = [2, 3, 4, 7, 8]
        X_ablated = X[:, indices]
        print(f"  Using only global centrality features ({X_ablated.shape[1]} features)")
        
    elif ablation_type == "degree_only":
        indices = [0, 6]
        X_ablated = X[:, indices]
        print(f"  Using only degree features ({X_ablated.shape[1]} features)")
        
    else:
        X_ablated = X.clone()
    
    return X_ablated


def apply_graph_structure_ablation(
    edge_index_mp: torch.Tensor,
    edge_index_full: torch.Tensor,
    is_obstetric: np.ndarray, 
    ablation_type: str,
    val_pos: np.ndarray,
    test_pos: np.ndarray,
    seed: int = 42
) -> torch.Tensor:
    """Apply graph structure ablation to message passing graph."""
    print(f"\nApplying graph structure ablation: {ablation_type}")
    
    if ablation_type in ["none", "no_community_flag", "local_only", "global_only", "degree_only"]:
        print(f"  Using leakage-safe message passing graph ({edge_index_mp.shape[1]} edges)")
        return edge_index_mp
    
    elif ablation_type == "cross_edges_only":
        mask = []
        for i in range(edge_index_mp.shape[1]):
            u, v = edge_index_mp[0, i].item(), edge_index_mp[1, i].item()
            is_cross = (is_obstetric[u] and not is_obstetric[v]) or \
                      (is_obstetric[v] and not is_obstetric[u])
            mask.append(is_cross)
        
        mask = torch.tensor(mask, dtype=torch.bool)
        edge_index_ablated = edge_index_mp[:, mask]
        print(f"  Using only cross-community edges: {edge_index_ablated.shape[1]} edges (was {edge_index_mp.shape[1]})")
        return edge_index_ablated
    
    elif ablation_type == "no_cross_edges_in_adj":
        mask = []
        for i in range(edge_index_mp.shape[1]):
            u, v = edge_index_mp[0, i].item(), edge_index_mp[1, i].item()
            is_within = (is_obstetric[u] and is_obstetric[v]) or \
                       (not is_obstetric[u] and not is_obstetric[v])
            mask.append(is_within)
        
        mask = torch.tensor(mask, dtype=torch.bool)
        edge_index_ablated = edge_index_mp[:, mask]
        print(f"  Using only within-community edges: {edge_index_ablated.shape[1]} edges (was {edge_index_mp.shape[1]})")
        return edge_index_ablated
    
    elif ablation_type == "rewired_graph":
        print(f"  Rewiring graph (degree-preserving)...")
        rng = np.random.RandomState(seed)
        
        # Create forbidden edge set
        forbidden_edges = set()
        for edge in val_pos:
            forbidden_edges.add((min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1]))))
        for edge in test_pos:
            forbidden_edges.add((min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1]))))
        
        print(f"  Forbidden edges (val+test): {len(forbidden_edges)}")
        
        edge_list = edge_index_mp.t().cpu().numpy()
        edge_set = set(map(tuple, edge_list))
        edge_list = list(edge_set)
        
        num_swaps = len(edge_list) * 5
        successful_swaps = 0
        rejected_swaps = 0
        
        for _ in range(num_swaps):
            if len(edge_list) < 2:
                break
            
            idx1, idx2 = rng.choice(len(edge_list), size=2, replace=False)
            u, v = edge_list[idx1]
            x, y = edge_list[idx2]
            
            if u != y and x != v:
                new_edge1 = (min(u, y), max(u, y))
                new_edge2 = (min(x, v), max(x, v))
                
                if (new_edge1 not in edge_set and new_edge2 not in edge_set and
                    new_edge1 not in forbidden_edges and new_edge2 not in forbidden_edges):
                    edge_set.remove((min(u, v), max(u, v)))
                    edge_set.remove((min(x, y), max(x, y)))
                    edge_list[idx1] = new_edge1
                    edge_list[idx2] = new_edge2
                    edge_set.add(new_edge1)
                    edge_set.add(new_edge2)
                    successful_swaps += 1
                elif new_edge1 in forbidden_edges or new_edge2 in forbidden_edges:
                    rejected_swaps += 1
        
        edge_array = np.array(list(edge_set), dtype=np.int64)
        edge_index_ablated = torch.tensor(edge_array.T, dtype=torch.long)
        print(f"  Rewired: {successful_swaps} successful swaps, {rejected_swaps} rejected (would leak), {edge_index_ablated.shape[1]} edges")
        return edge_index_ablated
    
    else:
        print(f"  Unknown ablation type, using leakage-safe MP graph")
        return edge_index_mp


def subsample_training_data(
    train_pos: np.ndarray, 
    train_neg: np.ndarray,
    train_fraction: float,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample training data to specified fraction."""
    if train_fraction >= 1.0:
        return train_pos, train_neg
    
    rng = np.random.RandomState(seed)
    n_pos = len(train_pos)
    n_new = int(n_pos * train_fraction)
    
    pos_indices = rng.choice(n_pos, size=n_new, replace=False)
    train_pos_sub = train_pos[pos_indices]
    
    neg_indices = rng.choice(len(train_neg), size=n_new, replace=False)
    train_neg_sub = train_neg[neg_indices]
    
    print(f"\nSubsampled training data to {train_fraction*100:.0f}%:")
    print(f"  Positive: {len(train_pos_sub)} (was {n_pos})")
    print(f"  Negative: {len(train_neg_sub)} (was {len(train_neg)})")
    
    return train_pos_sub, train_neg_sub


class KFoldAblationTrainer:
    """Trainer for k-fold ablation study."""
    
    def __init__(
        self,
        ablation_config: Dict,
        pkl_path: str,
        kfold_splits_path: str,
        hyperparams: Dict,
        max_epochs: int = 300,
        patience: int = 40,
        out_dir: str = 'ABLATIONS/kfold_runs',
        seed: int = 42
    ):
        self.ablation_config = ablation_config
        self.hyperparams = hyperparams
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ablation_name = ablation_config['name']
        self.run_dir = os.path.join(out_dir, f"ablation_{ablation_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"K-Fold Ablation: {ablation_name}")
        print(f"{'='*80}")
        print(f"Config: {ablation_config}")
        print(f"Output: {self.run_dir}")
        
        # Load graph
        print(f"\nLoading graph from {pkl_path}...")
        self.edge_index, self.x, self.num_nodes, self.is_obstetric, node_ids = load_graph(pkl_path)
        
        # Load concept names
        concept_csv_path = '../../data/EHRShot_sampled_2000patients/concept.csv'  # From ABLATIONS/ subdir
        if node_ids is not None and os.path.exists(concept_csv_path):
            concept_map = load_concept_names(concept_csv_path, node_ids)
            self.node_names = [concept_map.get(node_ids[i], node_ids[i]) for i in range(len(node_ids))]
        else:
            self.node_names = None
        
        # Load k-fold splits
        print(f"\nLoading k-fold splits from {kfold_splits_path}...")
        self.folds = load_kfold_splits(kfold_splits_path)
        print(f"Loaded {len(self.folds)} folds")
        
        self.device = get_device()
    
    def train_single_fold(self, fold_idx: int, fold_data: Dict) -> Dict:
        """Train on a single fold and return metrics."""
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*80}")
        
        set_seed(self.seed + fold_idx)  # Different seed per fold
        
        # Apply feature ablation
        x_ablated = apply_feature_ablation(self.x, self.is_obstetric, self.ablation_config['ablation_type'])
        
        # Store split data
        val_pos = fold_data['val_pos']
        val_neg = fold_data['val_neg']
        test_pos = fold_data['test_pos']
        test_neg = fold_data['test_neg']
        
        # Apply graph structure ablation
        edge_index_mp_safe = fold_data['edge_index_mp']
        edge_index_mp = apply_graph_structure_ablation(
            edge_index_mp_safe,
            self.edge_index,
            self.is_obstetric,
            self.ablation_config['ablation_type'],
            val_pos,
            test_pos,
            self.seed + fold_idx
        )
        
        # Apply training data subsampling
        train_pos_orig = fold_data['train_pos']
        train_neg_orig = fold_data['train_neg']
        
        train_pos, train_neg = subsample_training_data(
            train_pos_orig,
            train_neg_orig,
            self.ablation_config['train_fraction'],
            self.seed + fold_idx
        )
        
        # Move to device
        x_ablated = x_ablated.to(self.device)
        edge_index = self.edge_index.to(self.device)
        edge_index_mp = edge_index_mp.to(self.device)
        
        # Integrity checks
        print("\n" + "="*80)
        print(f"RUNNING SPLIT INTEGRITY CHECKS (Fold {fold_idx})")
        print("="*80)
        assert_split_integrity(
            train_pos, val_pos, test_pos,
            train_neg, val_neg, test_neg,
            edge_index_mp,
            edge_index,
            fold_id=fold_idx,
            is_obstetric=self.is_obstetric
        )
        print("="*80)
        print("✓ ALL INTEGRITY CHECKS PASSED")
        print("="*80)
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_pos)} pos, {len(train_neg)} neg")
        print(f"  Val: {len(val_pos)} pos, {len(val_neg)} neg")
        print(f"  Test: {len(test_pos)} pos, {len(test_neg)} neg")
        
        # Create model
        model = HybridGCNGraphSAGE(
            in_channels=x_ablated.shape[1],
            hidden_channels=self.hyperparams['hidden_channels'],
            out_channels=self.hyperparams['out_channels'],
            num_layers=self.hyperparams['num_layers'],
            gcn_dropout=self.hyperparams['dropout'],
            sage_dropout=self.hyperparams['dropout'],
            sage_aggr=self.hyperparams['sage_aggr'],
            fusion_method=self.hyperparams['fusion_method']
        ).to(self.device)
        
        print(f"\nModel parameters: {count_parameters(model):,}")
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.hyperparams['lr'],
            weight_decay=self.hyperparams['weight_decay']
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
        print(f"\nStarting training...")
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Patience: {self.patience}")
        print(f"  Selection metric: Validation AUROC")
        
        best_val_auroc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        best_metrics = None
        train_log = []
        best_train_probs = None
        best_val_probs = None
        best_test_probs = None
        
        # Create fold subdirectory
        fold_dir = os.path.join(self.run_dir, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        
        for epoch in range(self.max_epochs):
            model.train()
            optimizer.zero_grad()
            
            z = model(x_ablated, edge_index_mp)
            edge_scores = model.decode(z, train_edge_index)
            loss = F.binary_cross_entropy_with_logits(edge_scores, train_labels)
            
            loss.backward()
            optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                z = model(x_ablated, edge_index_mp)
                
                # Train metrics
                train_scores = model.decode(z, train_edge_index)
                train_probs = torch.sigmoid(train_scores).cpu().numpy()
                train_metrics = compute_metrics_dict(train_labels.cpu().numpy(), train_probs)
                
                # Val metrics
                val_scores = model.decode(z, val_edge_index)
                val_probs = torch.sigmoid(val_scores).cpu().numpy()
                val_metrics = compute_metrics_dict(val_labels.cpu().numpy(), val_probs)
                val_auroc = val_metrics['auroc']
                
                # Test metrics
                test_scores = model.decode(z, test_edge_index)
                test_probs = torch.sigmoid(test_scores).cpu().numpy()
                test_metrics = compute_metrics_dict(test_labels.cpu().numpy(), test_probs)
                
                # Log metrics
                log_entry = {
                    'epoch': epoch,
                    'loss': loss.item(),
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                    **{f'test_{k}': v for k, v in test_metrics.items()}
                }
                train_log.append(log_entry)
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUROC: {val_auroc:.4f} | Test AUROC: {test_metrics['auroc']:.4f} | Test AUPRC: {test_metrics['auprc']:.4f}")
                
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    best_epoch = epoch
                    epochs_no_improve = 0
                    
                    best_metrics = {
                        'train': train_metrics,
                        'val': val_metrics,
                        'test': test_metrics
                    }
                    
                    # Save best probabilities for persistence
                    best_train_probs = train_probs.copy()
                    best_val_probs = val_probs.copy()
                    best_test_probs = test_probs.copy()
                    
                    # Save checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_auroc': best_val_auroc,
                    }, os.path.join(fold_dir, 'best_checkpoint.pt'))
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch})")
                    break
        
        # Save training log
        log_path = os.path.join(fold_dir, 'train_log.csv')
        with open(log_path, 'w', newline='') as f:
            if train_log:
                writer = csv.DictWriter(f, fieldnames=train_log[0].keys())
                writer.writeheader()
                writer.writerows(train_log)
        
        # Save predictions for best model
        for split_name, edge_idx, labels, probs, pos_e, neg_e in [
            ('train', train_edge_index, train_labels, best_train_probs, train_pos, train_neg),
            ('val', val_edge_index, val_labels, best_val_probs, val_pos, val_neg),
            ('test', test_edge_index, test_labels, best_test_probs, test_pos, test_neg)
        ]:
            np.savez(
                os.path.join(fold_dir, f'{split_name}_predictions.npz'),
                y_true=labels.cpu().numpy(),
                y_pred_proba=probs,
                edge_index=edge_idx.cpu().numpy(),
                pos_edges=pos_e,
                neg_edges=neg_e
            )
        
        # Save top 20 predictions with concept names
        top20 = {
            'train': get_top_predictions(train_edge_index.cpu(), best_train_probs, train_labels.cpu().numpy(), self.node_names),
            'val': get_top_predictions(val_edge_index.cpu(), best_val_probs, val_labels.cpu().numpy(), self.node_names),
            'test': get_top_predictions(test_edge_index.cpu(), best_test_probs, test_labels.cpu().numpy(), self.node_names)
        }
        
        with open(os.path.join(fold_dir, 'top_20_predictions.json'), 'w') as f:
            json.dump(top20, f, indent=2)
        
        # Save fold summary
        fold_summary = {
            'fold_id': fold_idx,
            'best_epoch': best_epoch,
            'stopped_epoch': epoch if epochs_no_improve >= self.patience else self.max_epochs - 1,
            'comparison_metric': 'val_auroc',
            'comparison_value': best_val_auroc,
            'train_metrics': best_metrics['train'],
            'val_metrics': best_metrics['val'],
            'test_metrics': best_metrics['test']
        }
        
        with open(os.path.join(fold_dir, 'summary.json'), 'w') as f:
            json.dump(fold_summary, f, indent=2)
        
        print(f"\nFold {fold_idx} complete:")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Best val AUROC: {best_val_auroc:.4f}")
        print(f"  Test AUROC: {best_metrics['test']['auroc']:.4f}")
        print(f"  Test AUPRC: {best_metrics['test']['auprc']:.4f}")
        print(f"  Results saved to {fold_dir}")
        
        return best_metrics
    
    def train(self):
        """Train across all folds and aggregate results."""
        fold_results = []
        
        for fold_idx, fold_data in enumerate(self.folds):
            result = self.train_single_fold(fold_idx, fold_data)
            fold_results.append(result)
        
        # Aggregate results
        print(f"\n{'='*80}")
        print(f"AGGREGATING K-FOLD RESULTS")
        print(f"{'='*80}")
        
        metrics_summary = {}
        for split in ['train', 'val', 'test']:
            for metric in ['auroc', 'auprc', 'mcc', 'f1', 'accuracy']:
                values = [r[split][metric] for r in fold_results]
                metrics_summary[f'{split}_{metric}_mean'] = float(np.mean(values))
                metrics_summary[f'{split}_{metric}_std'] = float(np.std(values))
        
        # Print summary
        print(f"\n{self.ablation_config['name']} - K-Fold Results (n={len(self.folds)}):")
        print(f"\nValidation Metrics (mean ± std):")
        print(f"  AUROC: {metrics_summary['val_auroc_mean']:.4f} ± {metrics_summary['val_auroc_std']:.4f}")
        print(f"  AUPRC: {metrics_summary['val_auprc_mean']:.4f} ± {metrics_summary['val_auprc_std']:.4f}")
        print(f"  MCC:   {metrics_summary['val_mcc_mean']:.4f} ± {metrics_summary['val_mcc_std']:.4f}")
        print(f"\nTest Metrics (mean ± std):")
        print(f"  AUROC: {metrics_summary['test_auroc_mean']:.4f} ± {metrics_summary['test_auroc_std']:.4f}")
        print(f"  AUPRC: {metrics_summary['test_auprc_mean']:.4f} ± {metrics_summary['test_auprc_std']:.4f}")
        print(f"  MCC:   {metrics_summary['test_mcc_mean']:.4f} ± {metrics_summary['test_mcc_std']:.4f}")
        print(f"  F1:    {metrics_summary['test_f1_mean']:.4f} ± {metrics_summary['test_f1_std']:.4f}")
        
        # Save aggregated summary
        summary = {
            'ablation_name': self.ablation_config['name'],
            'ablation_config': self.ablation_config,
            'hyperparameters': self.hyperparams,
            'num_folds': len(self.folds),
            'metrics_summary': metrics_summary,
            'fold_results': fold_results,
            'comparison_metric': 'val_auroc',
            'comparison_value_mean': metrics_summary['val_auroc_mean'],
            'comparison_value_std': metrics_summary['val_auroc_std']
        }
        
        with open(os.path.join(self.run_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nAggregated results saved to {self.run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid GCN+GraphSAGE K-Fold Ablation Study")
    parser.add_argument('--ablation_type', type=str, required=True,
                       choices=['none', 'no_community_flag', 'local_only', 'global_only',
                               'degree_only', 'cross_edges_only', 'no_cross_edges_in_adj',
                               'rewired_graph'],
                       help='Type of ablation')
    parser.add_argument('--train_fraction', type=float, default=1.0,
                       help='Fraction of training data (0.25, 0.5, 0.75, 1.0)')
    parser.add_argument('--pkl', type=str, default='../../preprocessed_data/pregnancy_graph_data.pkl',
                       help='Path to graph pickle')
    parser.add_argument('--splits', type=str, default='../splits/kfold_k5_seed42.npz',
                       help='Path to k-fold split file')
    parser.add_argument('--hyperparams_file', type=str, 
                       default='../hyperparams/hybrid_gcn_graphsage_best.json',
                       help='Path to best hyperparameters')
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--out_dir', type=str, default='ABLATIONS/kfold_runs')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Load hyperparameters
    with open(args.hyperparams_file, 'r') as f:
        hyperparams = json.load(f)
    
    # Create ablation config
    ablation_config = {
        'name': f"{args.ablation_type}_frac{args.train_fraction:.2f}",
        'ablation_type': args.ablation_type,
        'train_fraction': args.train_fraction
    }
    
    # Train
    trainer = KFoldAblationTrainer(
        ablation_config=ablation_config,
        pkl_path=args.pkl,
        kfold_splits_path=args.splits,
        hyperparams=hyperparams,
        max_epochs=args.max_epochs,
        patience=args.patience,
        out_dir=args.out_dir,
        seed=args.seed
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
