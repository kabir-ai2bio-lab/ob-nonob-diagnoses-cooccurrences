"""
Nested k-fold hyperparameter tuning with strict experimental protocol.

STEP 2: Nested cross-validation
- Outer loop: 5-fold (for final performance estimation)
- Inner loop: 3-fold (for hyperparameter selection)
- No outer test leakage during tuning
- Select hyperparameters by mean inner validation AUROC
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_io import load_graph
from split_manager import load_kfold_splits, _sample_negative_edges, _construct_mp_edge_index
from models import GCN, GraphSAGE, GAT, HybridGCNGAT, HybridGCNGraphSAGE, GraphSAGEGATHybrid
from utils import set_seed, get_device, count_parameters
from metrics import compute_metrics
from split_asserts import assert_split_integrity


class HyperparamTuner:
    """Nested k-fold hyperparameter tuner."""
    
    def __init__(
        self,
        pkl_path: str,
        splits_path: str,
        model_name: str,
        param_grid: Dict[str, List[Any]],
        inner_k: int = 3,
        seed: int = 42,
        max_epochs: int = 300,
        patience: int = 40,
        out_dir: str = 'runs/hyperparam_tuning'
    ):
        self.pkl_path = pkl_path
        self.splits_path = splits_path
        self.model_name = model_name
        self.param_grid = param_grid
        self.inner_k = inner_k
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.out_dir = out_dir
        
        # Load graph
        print(f"[DEBUG] Loading graph from {pkl_path}...", flush=True)
        self.edge_index, self.x, self.num_nodes, self.is_obstetric = load_graph(pkl_path)
        print(f"[DEBUG] Graph loaded: {self.num_nodes} nodes", flush=True)
        
        print(f"[DEBUG] Getting device...", flush=True)
        self.device = get_device()
        print(f"[DEBUG] Device: {self.device}", flush=True)
        
        # Load pre-saved k-fold splits
        print(f"[DEBUG] Loading pre-saved k-fold splits from {splits_path}...", flush=True)
        self.folds = load_kfold_splits(splits_path)
        self.outer_k = len(self.folds)
        
        print(f"[DEBUG] Loaded {self.outer_k} outer folds", flush=True)
        
        # Move to device
        print(f"[DEBUG] Moving data to device...", flush=True)
        self.x = self.x.to(self.device)
        self.edge_index = self.edge_index.to(self.device)
        print(f"[DEBUG] Data moved to {self.device}", flush=True)
        
        os.makedirs(out_dir, exist_ok=True)
        print(f"[DEBUG] Initialization complete!", flush=True)
    
    def run(self):
        """Run nested k-fold hyperparameter tuning."""
        print(f"[DEBUG] run() method called", flush=True)
        print(f"\n{'='*80}", flush=True)
        print(f"Nested K-Fold Hyperparameter Tuning: {self.model_name}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Outer K: {self.outer_k}, Inner K: {self.inner_k}", flush=True)
        print(f"Seed: {self.seed}", flush=True)
        print(f"Max epochs: {self.max_epochs}, Patience: {self.patience}", flush=True)
        print(flush=True)
        
        print(f"[DEBUG] Starting outer fold loop...", flush=True)
        outer_results = []
        
        for fold_data in self.folds:
            outer_fold = fold_data['fold_id']
            
            print(f"\n{'#'*80}", flush=True)
            print(f"OUTER FOLD {outer_fold}/{self.outer_k - 1}", flush=True)
            print(f"{'#'*80}", flush=True)
            
            # Extract pre-saved outer fold data
            print(f"[DEBUG] Extracting outer fold data...", flush=True)
            outer_train_pos = fold_data['train_pos']
            outer_val_pos = fold_data['val_pos']
            outer_test_pos = fold_data['test_pos']
            outer_train_neg = fold_data['train_neg']
            outer_val_neg = fold_data['val_neg']
            outer_test_neg = fold_data['test_neg']
            print(f"[DEBUG] Extracted: train_pos={len(outer_train_pos)}, val_pos={len(outer_val_pos)}, test_pos={len(outer_test_pos)}", flush=True)
            
            # Combine train+val for inner tuning
            print(f"[DEBUG] Combining train+val for inner CV...", flush=True)
            outer_trainval_pos = np.vstack([outer_train_pos, outer_val_pos])
            outer_trainval_neg = np.vstack([outer_train_neg, outer_val_neg])
            
            print(f"Outer trainval size: {len(outer_trainval_pos)} pos, {len(outer_trainval_neg)} neg", flush=True)
            print(f"Outer test size: {len(outer_test_pos)} pos, {len(outer_test_neg)} neg (HELD OUT)", flush=True)
            
            # Inner k-fold on outer_trainval
            print(f"[DEBUG] Calling _tune_on_trainval...", flush=True)
            best_params, best_inner_auroc = self._tune_on_trainval(
                outer_trainval_pos, outer_trainval_neg, outer_fold
            )
            
            print(f"\nOuter fold {outer_fold} best params: {best_params}")
            print(f"Best inner validation AUROC: {best_inner_auroc:.4f}")
            
            # Retrain on full outer trainval with best params and evaluate on outer test
            outer_test_metrics = self._retrain_and_evaluate(
                outer_trainval_pos, outer_trainval_neg,
                outer_test_pos, outer_test_neg,
                best_params, outer_fold
            )
            
            outer_results.append({
                'outer_fold': outer_fold,
                'best_params': best_params,
                'best_inner_auroc': float(best_inner_auroc),
                'outer_test_metrics': outer_test_metrics
            })
        
        # Aggregate results
        self._save_results(outer_results)
        
        print(f"\n{'='*80}")
        print("Nested K-Fold Tuning Complete")
        print(f"{'='*80}")
    
    def _tune_on_trainval(self, trainval_pos: np.ndarray, trainval_neg: np.ndarray, outer_fold: int):
        """
        Tune hyperparameters using inner k-fold on trainval set.
        
        Returns:
            best_params, best_mean_auroc
        """
        print(f"\n[DEBUG] _tune_on_trainval called for outer fold {outer_fold}", flush=True)
        print(f"Inner k-fold tuning on outer fold {outer_fold}...", flush=True)
        
        # Generate all hyperparameter combinations
        print(f"[DEBUG] Generating parameter combinations...", flush=True)
        param_combos = self._generate_param_combinations()
        
        print(f"Total hyperparameter combinations: {len(param_combos)}", flush=True)
        
        print(f"[DEBUG] Starting parameter search loop...", flush=True)
        best_params = None
        best_mean_auroc = -1.0
        
        print(f"[DEBUG] Entering combo loop with {len(param_combos)} combos...", flush=True)
        for combo_idx, params in enumerate(param_combos):
            print(f"\n[DEBUG] Combo {combo_idx+1} starting...", flush=True)
            print(f"\n--- Combo {combo_idx+1}/{len(param_combos)}: {params} ---", flush=True)
            
            # Inner k-fold validation
            inner_aurocs = []
            
            inner_kfold = KFold(n_splits=self.inner_k, shuffle=True, random_state=self.seed + 999)
            
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kfold.split(trainval_pos)):
                inner_train_pos = trainval_pos[inner_train_idx]
                inner_val_pos = trainval_pos[inner_val_idx]
                inner_train_neg = trainval_neg[inner_train_idx]
                inner_val_neg = trainval_neg[inner_val_idx]
                
                # Construct edge_index_mp (remove val edges)
                edge_index_mp = _construct_mp_edge_index(
                    self.edge_index.cpu(), inner_val_pos, np.array([])  # No test in inner loop
                )
                edge_index_mp = edge_index_mp.to(self.device)
                
                # SAFETY CHECK: Assert split integrity before training
                assert_split_integrity(
                    inner_train_pos, inner_val_pos, np.array([]),
                    inner_train_neg, inner_val_neg, np.array([]),
                    edge_index_mp, self.edge_index,
                    fold_id=f"outer_{outer_fold}_inner_{inner_fold}"
                )
                
                # Train and evaluate
                val_auroc = self._train_single_config(
                    inner_train_pos, inner_train_neg,
                    inner_val_pos, inner_val_neg,
                    edge_index_mp, params, inner_fold
                )
                
                inner_aurocs.append(val_auroc)
                print(f"  Inner fold {inner_fold}: val_auroc={val_auroc:.4f}")
            
            mean_auroc = np.mean(inner_aurocs)
            std_auroc = np.std(inner_aurocs)
            
            print(f"Mean inner AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
            
            if mean_auroc > best_mean_auroc:
                best_mean_auroc = mean_auroc
                best_params = params
                print(f"*** NEW BEST MEAN AUROC: {best_mean_auroc:.4f} ***")
        
        return best_params, best_mean_auroc
    
    def _retrain_and_evaluate(
        self, trainval_pos: np.ndarray, trainval_neg: np.ndarray,
        test_pos: np.ndarray, test_neg: np.ndarray,
        best_params: Dict, outer_fold: int
    ):
        """
        Retrain on full trainval with best params, evaluate on outer test.
        Uses pre-saved negatives from the outer fold.
        """
        print(f"\nRetraining with best params on full outer trainval...")
        
        # Construct edge_index_mp (remove test edges)
        edge_index_mp = _construct_mp_edge_index(
            self.edge_index.cpu(), np.array([]), test_pos  # No val, only test
        )
        edge_index_mp = edge_index_mp.to(self.device)
        
        # Train model
        model = self._create_model(best_params).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params.get('weight_decay', 0.0))
        
        best_trainval_auroc = 0.0
        epochs_no_improve = 0
        
        for epoch in range(self.max_epochs):
            model.train()
            
            # Training step
            optimizer.zero_grad()
            z = model(self.x, edge_index_mp)
            
            # Prepare training edges
            train_pos_tensor = torch.tensor(trainval_pos, dtype=torch.long).t().to(self.device)
            train_neg_tensor = torch.tensor(trainval_neg, dtype=torch.long).t().to(self.device)
            
            train_edge_index = torch.cat([train_pos_tensor, train_neg_tensor], dim=1)
            train_labels = torch.cat([
                torch.ones(train_pos_tensor.shape[1]),
                torch.zeros(train_neg_tensor.shape[1])
            ]).to(self.device)
            
            scores = model.decode(z, train_edge_index)
            loss = F.binary_cross_entropy_with_logits(scores, train_labels)
            
            loss.backward()
            optimizer.step()
            
            # Validation on trainval (for early stopping)
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    z = model(self.x, edge_index_mp)
                    scores = model.decode(z, train_edge_index)
                    probs = torch.sigmoid(scores).cpu().numpy()
                    
                    trainval_auroc = compute_metrics(train_labels.cpu().numpy(), probs)['auroc']
                    
                    if trainval_auroc > best_trainval_auroc:
                        best_trainval_auroc = trainval_auroc
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    
                    if epochs_no_improve >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        # Evaluate on outer test
        model.eval()
        with torch.no_grad():
            z = model(self.x, edge_index_mp)
            
            test_pos_tensor = torch.tensor(test_pos, dtype=torch.long).t().to(self.device)
            test_neg_tensor = torch.tensor(test_neg, dtype=torch.long).t().to(self.device)
            test_edge_index = torch.cat([test_pos_tensor, test_neg_tensor], dim=1)
            test_labels = torch.cat([
                torch.ones(test_pos_tensor.shape[1]),
                torch.zeros(test_neg_tensor.shape[1])
            ]).to(self.device)
            
            scores = model.decode(z, test_edge_index)
            probs = torch.sigmoid(scores).cpu().numpy()
            
            test_metrics = compute_metrics(test_labels.cpu().numpy(), probs)
        
        print(f"Outer test metrics: {test_metrics}")
        
        return {k: float(v) for k, v in test_metrics.items()}
    
    def _train_single_config(
        self, train_pos, train_neg, val_pos, val_neg,
        edge_index_mp, params, fold_id
    ):
        """Train a single model configuration and return validation AUROC."""
        set_seed(self.seed + fold_id)
        
        model = self._create_model(params).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params.get('weight_decay', 0.0))
        
        # Prepare data
        train_pos_tensor = torch.tensor(train_pos, dtype=torch.long).t().to(self.device)
        train_neg_tensor = torch.tensor(train_neg, dtype=torch.long).t().to(self.device)
        val_pos_tensor = torch.tensor(val_pos, dtype=torch.long).t().to(self.device)
        val_neg_tensor = torch.tensor(val_neg, dtype=torch.long).t().to(self.device)
        
        train_edge_index = torch.cat([train_pos_tensor, train_neg_tensor], dim=1)
        train_labels = torch.cat([
            torch.ones(train_pos_tensor.shape[1]),
            torch.zeros(train_neg_tensor.shape[1])
        ]).to(self.device)
        
        val_edge_index = torch.cat([val_pos_tensor, val_neg_tensor], dim=1)
        val_labels = torch.cat([
            torch.ones(val_pos_tensor.shape[1]),
            torch.zeros(val_neg_tensor.shape[1])
        ]).to(self.device)
        
        best_val_auroc = 0.0
        epochs_no_improve = 0
        
        for epoch in range(self.max_epochs):
            model.train()
            optimizer.zero_grad()
            
            z = model(self.x, edge_index_mp)
            scores = model.decode(z, train_edge_index)
            loss = F.binary_cross_entropy_with_logits(scores, train_labels)
            
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    z = model(self.x, edge_index_mp)
                    val_scores = model.decode(z, val_edge_index)
                    val_probs = torch.sigmoid(val_scores).cpu().numpy()
                    
                    val_auroc = compute_metrics(val_labels.cpu().numpy(), val_probs)['auroc']
                    
                    if val_auroc > best_val_auroc:
                        best_val_auroc = val_auroc
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    
                    if epochs_no_improve >= self.patience // 4:  # Reduced patience for inner loop
                        break
        
        return best_val_auroc
    
    def _create_model(self, params: Dict):
        """Create model instance from parameters."""
        in_channels = self.x.shape[1]
        
        if self.model_name == 'gcn':
            return GCN(
                in_channels=in_channels,
                hidden_channels=params['hidden_channels'],
                out_channels=params['out_channels'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        elif self.model_name == 'graphsage':
            return GraphSAGE(
                in_channels=in_channels,
                hidden_channels=params['hidden_channels'],
                out_channels=params['out_channels'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                aggr=params.get('aggr', 'mean')
            )
        elif self.model_name == 'gat':
            return GAT(
                in_channels=in_channels,
                hidden_channels=params['hidden_channels'],
                out_channels=params['out_channels'],
                num_layers=params['num_layers'],
                num_heads=params.get('num_heads', 4),
                dropout=params['dropout']
            )
        elif self.model_name == 'hybrid_gcn_gat':
            return HybridGCNGAT(
                in_channels=in_channels,
                hidden_channels=params['hidden_channels'],
                out_channels=params['out_channels'],
                num_layers=params['num_layers'],
                gcn_dropout=params.get('gcn_dropout', params['dropout']),
                gat_dropout=params.get('gat_dropout', params['dropout']),
                gat_heads=params.get('gat_heads', 4),
                fusion_method=params.get('fusion_method', 'projected_concat')
            )
        elif self.model_name == 'hybrid_gcn_graphsage':
            return HybridGCNGraphSAGE(
                in_channels=in_channels,
                hidden_channels=params['hidden_channels'],
                out_channels=params['out_channels'],
                num_layers=params['num_layers'],
                gcn_dropout=params.get('gcn_dropout', params['dropout']),
                sage_dropout=params.get('sage_dropout', params['dropout']),
                sage_aggr=params.get('sage_aggr', 'mean'),
                fusion_method=params.get('fusion_method', 'projected_concat')
            )
        elif self.model_name == 'graphsage_gat_hybrid':
            return GraphSAGEGATHybrid(
                in_channels=in_channels,
                sage_hidden_channels=params.get('sage_hidden_channels', 128),
                sage_num_layers=params.get('sage_num_layers', 2),
                sage_aggr=params.get('sage_aggr', 'mean'),
                dropout=params.get('sage_dropout', params.get('dropout', 0.3)),
                gat_num_layers=params.get('gat_num_layers', 1),
                gat_hidden_channels=params.get('gat_hidden_channels', 64),
                gat_heads=params.get('gat_num_heads', 4),
                gat_dropout=params.get('gat_dropout', params.get('dropout', 0.3)),
                out_channels=params['out_channels']
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _generate_param_combinations(self):
        """Generate all hyperparameter combinations from grid."""
        import itertools
        
        print(f"[DEBUG _generate_param_combinations] Starting...", flush=True)
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        
        print(f"[DEBUG _generate_param_combinations] Keys: {keys}, generating products...", flush=True)
        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        
        print(f"[DEBUG _generate_param_combinations] Generated {len(combinations)} combinations", flush=True)
        return combinations
    
    def _save_results(self, outer_results: List[Dict]):
        """Save tuning results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.out_dir, f"{self.model_name}_nested_kfold_{timestamp}.json")
        
        # Aggregate outer test metrics
        all_auroc = [r['outer_test_metrics']['auroc'] for r in outer_results]
        all_auprc = [r['outer_test_metrics']['auprc'] for r in outer_results]
        
        summary = {
            'model_name': self.model_name,
            'outer_k': self.outer_k,
            'inner_k': self.inner_k,
            'seed': self.seed,
            'timestamp': timestamp,
            'param_grid': self.param_grid,
            'outer_results': outer_results,
            'aggregated_metrics': {
                'mean_test_auroc': float(np.mean(all_auroc)),
                'std_test_auroc': float(np.std(all_auroc)),
                'mean_test_auprc': float(np.mean(all_auprc)),
                'std_test_auprc': float(np.std(all_auprc))
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        print(f"\nAggregated Test AUROC: {summary['aggregated_metrics']['mean_test_auroc']:.4f} ± {summary['aggregated_metrics']['std_test_auroc']:.4f}")
        print(f"Aggregated Test AUPRC: {summary['aggregated_metrics']['mean_test_auprc']:.4f} ± {summary['aggregated_metrics']['std_test_auprc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Nested k-fold hyperparameter tuning")
    parser.add_argument('--pkl', type=str, required=True, help='Path to graph pickle')
    parser.add_argument('--splits', type=str, required=True, help='Path to pre-saved k-fold splits NPZ')
    parser.add_argument('--model', type=str, required=True,
                        choices=['gcn', 'graphsage', 'gat', 'hybrid_gcn_gat', 'hybrid_gcn_graphsage', 'graphsage_gat_hybrid'],
                        help='Model name')
    parser.add_argument('--param_grid_file', type=str, required=True,
                        help='Path to JSON file with hyperparameter grid')
    parser.add_argument('--inner_k', type=int, default=3, help='Inner folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_epochs', type=int, default=300, help='Max epochs per config')
    parser.add_argument('--patience', type=int, default=40, help='Early stopping patience')
    parser.add_argument('--out_dir', type=str, default='runs/hyperparam_tuning', help='Output directory')
    
    args = parser.parse_args()
    
    # Load parameter grid
    print(f"[DEBUG MAIN] Loading parameter grid from {args.param_grid_file}...", flush=True)
    with open(args.param_grid_file, 'r') as f:
        param_grid = json.load(f)
    print(f"[DEBUG MAIN] Loaded {len(param_grid)} parameters", flush=True)
    
    # Run tuning
    print(f"[DEBUG MAIN] Creating HyperparamTuner...", flush=True)
    tuner = HyperparamTuner(
        pkl_path=args.pkl,
        splits_path=args.splits,
        model_name=args.model,
        param_grid=param_grid,
        inner_k=args.inner_k,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        out_dir=args.out_dir
    )
    
    print(f"[DEBUG MAIN] Calling tuner.run()...", flush=True)
    tuner.run()
    print(f"[DEBUG MAIN] tuner.run() completed!", flush=True)


if __name__ == '__main__':
    main()
