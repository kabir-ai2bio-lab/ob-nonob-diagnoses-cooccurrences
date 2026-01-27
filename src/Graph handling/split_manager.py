"""
Split manager for generating and loading train/val/test splits with strict leakage prevention.

STEP 1.2-1.5: Single and k-fold split generation with edge_index_mp construction.
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Tuple, Dict, List
import numpy as np
import torch
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_io import load_graph


def generate_single_split(
    pkl_path: str,
    out_dir: str = "splits",
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    overwrite: bool = False
) -> Dict:
    """
    Generate a single train/val/test split with strict leakage prevention.
    
    Args:
        pkl_path: Path to graph pickle file
        out_dir: Output directory for splits
        seed: Random seed for reproducibility
        ratios: (train_ratio, val_ratio, test_ratio)
        overwrite: If False and split exists, load it instead of regenerating
        
    Returns:
        Dictionary containing all split components
    """
    os.makedirs(out_dir, exist_ok=True)
    
    npz_path = os.path.join(out_dir, f"single_seed{seed}.npz")
    json_path = os.path.join(out_dir, f"single_seed{seed}.json")
    
    # Check if split already exists
    if os.path.exists(npz_path) and not overwrite:
        print(f"Split already exists at {npz_path}, loading...")
        return load_single_split(npz_path)
    
    print(f"\nGenerating single split with seed={seed}, ratios={ratios}")
    
    # Load graph
    edge_index, x, num_nodes, is_obstetric = load_graph(pkl_path)
    
    # Define cross-community positive edges
    print("\nDefining cross-community link prediction universe...")
    pos_edges = _get_cross_community_edges(edge_index, is_obstetric)
    
    num_pos = len(pos_edges)
    print(f"Total cross-community positive edges: {num_pos}")
    
    # Split positives
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    rng = np.random.RandomState(seed)
    indices = np.arange(num_pos)
    rng.shuffle(indices)
    
    n_train = int(num_pos * train_ratio)
    n_val = int(num_pos * val_ratio)
    
    train_pos_idx = indices[:n_train]
    val_pos_idx = indices[n_train:n_train + n_val]
    test_pos_idx = indices[n_train + n_val:]
    
    train_pos = pos_edges[train_pos_idx]
    val_pos = pos_edges[val_pos_idx]
    test_pos = pos_edges[test_pos_idx]
    
    print(f"Positive splits: train={len(train_pos)}, val={len(val_pos)}, test={len(test_pos)}")
    
    # Sample balanced negatives (deterministic)
    print("\nSampling balanced negative edges...")
    sampled_negatives = set()  # Track all sampled negatives to prevent overlap
    
    train_neg = _sample_negative_edges(
        num_nodes, is_obstetric, edge_index, len(train_pos), seed, offset=0,
        exclude_edges=sampled_negatives
    )
    sampled_negatives.update((min(e[0], e[1]), max(e[0], e[1])) for e in train_neg)
    
    val_neg = _sample_negative_edges(
        num_nodes, is_obstetric, edge_index, len(val_pos), seed, offset=len(train_pos),
        exclude_edges=sampled_negatives
    )
    sampled_negatives.update((min(e[0], e[1]), max(e[0], e[1])) for e in val_neg)
    
    test_neg = _sample_negative_edges(
        num_nodes, is_obstetric, edge_index, len(test_pos), seed, offset=len(train_pos) + len(val_pos),
        exclude_edges=sampled_negatives
    )
    
    print(f"Negative splits: train={len(train_neg)}, val={len(val_neg)}, test={len(test_neg)}")
    
    # Construct edge_index_mp (message-passing graph with leakage prevention)
    print("\nConstructing edge_index_mp (leakage prevention)...")
    edge_index_mp = _construct_mp_edge_index(edge_index, val_pos, test_pos)
    
    # Leakage check
    _check_no_leakage(edge_index_mp, val_pos, test_pos, "single split")
    
    # Save splits
    np.savez(
        npz_path,
        train_pos=train_pos,
        train_neg=train_neg,
        val_pos=val_pos,
        val_neg=val_neg,
        test_pos=test_pos,
        test_neg=test_neg,
        edge_index_mp=edge_index_mp.numpy(),
        is_obstetric=is_obstetric.numpy()
    )
    
    # Save metadata
    metadata = {
        'seed': seed,
        'ratios': ratios,
        'counts': {
            'train_pos': len(train_pos),
            'train_neg': len(train_neg),
            'val_pos': len(val_pos),
            'val_neg': len(val_neg),
            'test_pos': len(test_pos),
            'test_neg': len(test_neg)
        },
        'negative_sampling_strategy': 'balanced_deterministic',
        'timestamp': datetime.now().isoformat(),
        'pkl_path': pkl_path,
        'num_nodes': num_nodes,
        'num_edges_full': edge_index.shape[1],
        'num_edges_mp': edge_index_mp.shape[1]
    }
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSplit saved to {npz_path}")
    print(f"Metadata saved to {json_path}")
    
    return load_single_split(npz_path)


def generate_kfold_splits(
    pkl_path: str,
    out_dir: str = "splits",
    seed: int = 42,
    k: int = 5,
    val_ratio_within_train: float = 0.15,
    overwrite: bool = False
) -> List[Dict]:
    """
    Generate k-fold cross-validation splits with nested validation sets.
    
    Uses sklearn.model_selection.KFold to split positive edges ONLY.
    For each fold, creates balanced negatives deterministically.
    
    Args:
        pkl_path: Path to graph pickle file
        out_dir: Output directory
        seed: Random seed
        k: Number of folds
        val_ratio_within_train: Proportion of training data for validation
        overwrite: Whether to overwrite existing splits
        
    Returns:
        List of k dictionaries, one per fold
    """
    os.makedirs(out_dir, exist_ok=True)
    
    npz_path = os.path.join(out_dir, f"kfold_k{k}_seed{seed}.npz")
    json_path = os.path.join(out_dir, f"kfold_k{k}_seed{seed}.json")
    
    # Check if splits already exist
    if os.path.exists(npz_path) and not overwrite:
        print(f"K-fold splits already exist at {npz_path}, loading...")
        return load_kfold_splits(npz_path)
    
    print(f"\nGenerating {k}-fold splits with seed={seed}")
    print(f"Val ratio within train: {val_ratio_within_train}")
    
    # Load graph
    edge_index, x, num_nodes, is_obstetric = load_graph(pkl_path)
    
    # Define cross-community positive edges
    print("\nDefining cross-community link prediction universe...")
    pos_edges = _get_cross_community_edges(edge_index, is_obstetric)
    
    num_pos = len(pos_edges)
    print(f"Total cross-community positive edges: {num_pos}")
    
    # Initialize KFold
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    fold_data = {}
    
    for fold_idx, (trainval_indices, test_indices) in enumerate(kfold.split(pos_edges)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}")
        print(f"{'='*60}")
        
        # Test positives
        test_pos = pos_edges[test_indices]
        
        # Train+Val positives
        trainval_pos = pos_edges[trainval_indices]
        
        # Split trainval into train and val
        # Use fold-specific deterministic RNG
        trainval_rng = np.random.RandomState(seed * 10000 + fold_idx)
        trainval_perm = trainval_rng.permutation(len(trainval_pos))
        
        n_val = int(len(trainval_pos) * val_ratio_within_train)
        
        val_pos_idx = trainval_perm[:n_val]
        train_pos_idx = trainval_perm[n_val:]
        
        train_pos = trainval_pos[train_pos_idx]
        val_pos = trainval_pos[val_pos_idx]
        
        print(f"Positive splits: train={len(train_pos)}, val={len(val_pos)}, test={len(test_pos)}")
        
        # Sample balanced negatives (deterministic using seed and fold_idx)
        fold_seed_base = seed * 10000 + fold_idx
        
        sampled_negatives = set()  # Track all sampled negatives to prevent overlap
        
        train_neg = _sample_negative_edges(
            num_nodes, is_obstetric, edge_index, len(train_pos), fold_seed_base, offset=0,
            exclude_edges=sampled_negatives
        )
        sampled_negatives.update((min(e[0], e[1]), max(e[0], e[1])) for e in train_neg)
        
        val_neg = _sample_negative_edges(
            num_nodes, is_obstetric, edge_index, len(val_pos), fold_seed_base, offset=len(train_pos),
            exclude_edges=sampled_negatives
        )
        sampled_negatives.update((min(e[0], e[1]), max(e[0], e[1])) for e in val_neg)
        
        test_neg = _sample_negative_edges(
            num_nodes, is_obstetric, edge_index, len(test_pos), fold_seed_base, offset=len(train_pos) + len(val_pos),
            exclude_edges=sampled_negatives
        )
        
        print(f"Negative splits: train={len(train_neg)}, val={len(val_neg)}, test={len(test_neg)}")
        
        # Construct edge_index_mp
        print("Constructing edge_index_mp...")
        edge_index_mp = _construct_mp_edge_index(edge_index, val_pos, test_pos)
        
        # Leakage check
        _check_no_leakage(edge_index_mp, val_pos, test_pos, f"fold {fold_idx}")
        
        # Store fold data
        fold_data[f'fold_{fold_idx}_train_pos'] = train_pos
        fold_data[f'fold_{fold_idx}_train_neg'] = train_neg
        fold_data[f'fold_{fold_idx}_val_pos'] = val_pos
        fold_data[f'fold_{fold_idx}_val_neg'] = val_neg
        fold_data[f'fold_{fold_idx}_test_pos'] = test_pos
        fold_data[f'fold_{fold_idx}_test_neg'] = test_neg
        fold_data[f'fold_{fold_idx}_edge_index_mp'] = edge_index_mp.numpy()
    
    # Add metadata
    fold_data['is_obstetric'] = is_obstetric.numpy()
    fold_data['k'] = k
    fold_data['seed'] = seed
    
    # Save
    np.savez(npz_path, **fold_data)
    
    # Save JSON metadata
    metadata = {
        'k': k,
        'seed': seed,
        'val_ratio_within_train': val_ratio_within_train,
        'negative_sampling_strategy': 'balanced_deterministic_per_fold',
        'timestamp': datetime.now().isoformat(),
        'pkl_path': pkl_path,
        'num_nodes': num_nodes,
        'num_edges_full': edge_index.shape[1],
        'folds': []
    }
    
    for fold_idx in range(k):
        fold_meta = {
            'fold_id': fold_idx,
            'train_pos': int(np.sum([f'fold_{fold_idx}_train_pos' in fold_data])),
            'val_pos': int(np.sum([f'fold_{fold_idx}_val_pos' in fold_data])),
            'test_pos': int(np.sum([f'fold_{fold_idx}_test_pos' in fold_data]))
        }
        metadata['folds'].append(fold_meta)
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nK-fold splits saved to {npz_path}")
    print(f"Metadata saved to {json_path}")
    
    return load_kfold_splits(npz_path)


def load_single_split(npz_path: str) -> Dict:
    """Load a single split from .npz file."""
    data = np.load(npz_path)
    
    return {
        'train_pos': data['train_pos'],
        'train_neg': data['train_neg'],
        'val_pos': data['val_pos'],
        'val_neg': data['val_neg'],
        'test_pos': data['test_pos'],
        'test_neg': data['test_neg'],
        'edge_index_mp': torch.tensor(data['edge_index_mp'], dtype=torch.long),
        'is_obstetric': torch.tensor(data['is_obstetric'], dtype=torch.bool)
    }


def load_kfold_splits(npz_path: str) -> List[Dict]:
    """
    Load k-fold splits from .npz file.
    
    Returns:
        List of dictionaries, one per fold
    """
    data = np.load(npz_path)
    
    k = int(data['k'])
    is_obstetric = torch.tensor(data['is_obstetric'], dtype=torch.bool)
    
    folds = []
    
    for fold_idx in range(k):
        fold_dict = {
            'fold_id': fold_idx,
            'train_pos': data[f'fold_{fold_idx}_train_pos'],
            'train_neg': data[f'fold_{fold_idx}_train_neg'],
            'val_pos': data[f'fold_{fold_idx}_val_pos'],
            'val_neg': data[f'fold_{fold_idx}_val_neg'],
            'test_pos': data[f'fold_{fold_idx}_test_pos'],
            'test_neg': data[f'fold_{fold_idx}_test_neg'],
            'edge_index_mp': torch.tensor(data[f'fold_{fold_idx}_edge_index_mp'], dtype=torch.long),
            'is_obstetric': is_obstetric
        }
        folds.append(fold_dict)
    
    return folds


def _get_cross_community_edges(edge_index: torch.LongTensor, is_obstetric: torch.Tensor) -> np.ndarray:
    """
    Extract cross-community edges (u,v) where is_obstetric[u] != is_obstetric[v].
    
    Returns:
        numpy array of shape (num_edges, 2)
    """
    u = edge_index[0]
    v = edge_index[1]
    
    # Cross-community mask
    cross_community_mask = is_obstetric[u] != is_obstetric[v]
    
    cross_edges = edge_index[:, cross_community_mask].t().numpy()
    
    return cross_edges


def _sample_negative_edges(
    num_nodes: int,
    is_obstetric: torch.Tensor,
    edge_index: torch.LongTensor,
    num_samples: int,
    seed: int,
    offset: int = 0,
    exclude_edges: set = None
) -> np.ndarray:
    """
    Sample cross-community negative edges deterministically.
    
    Negatives are non-existent edges between obstetric and non-obstetric nodes.
    
    Args:
        num_nodes: Total number of nodes
        is_obstetric: Boolean tensor indicating obstetric community
        edge_index: Full graph edge index
        num_samples: Number of negatives to sample
        seed: Random seed
        offset: Offset for seed derivation (ensures different negatives per split)
        exclude_edges: Set of edges to exclude (for preventing overlap across splits)
        
    Returns:
        Array of shape (num_samples, 2)
    """
    rng = np.random.RandomState(seed + offset)
    
    if exclude_edges is None:
        exclude_edges = set()
    
    # Create set of existing edges for fast lookup
    edge_set = set()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        edge_set.add((min(u, v), max(u, v)))
    
    # Merge with exclude_edges
    forbidden_edges = edge_set | exclude_edges
    
    # Get node lists
    ob_nodes = torch.where(is_obstetric)[0].numpy()
    non_ob_nodes = torch.where(~is_obstetric)[0].numpy()
    
    if len(ob_nodes) == 0 or len(non_ob_nodes) == 0:
        raise RuntimeError("Cannot sample cross-community negatives: one community is empty")
    
    negatives = []
    max_attempts = num_samples * 100
    attempts = 0
    
    while len(negatives) < num_samples and attempts < max_attempts:
        u = rng.choice(ob_nodes)
        v = rng.choice(non_ob_nodes)
        
        edge = (min(u, v), max(u, v))
        
        if edge not in forbidden_edges and edge not in negatives:
            negatives.append(edge)
        
        attempts += 1
    
    if len(negatives) < num_samples:
        raise RuntimeError(
            f"Could only sample {len(negatives)}/{num_samples} negative edges after {attempts} attempts"
        )
    
    return np.array(negatives, dtype=np.int64)


def _construct_mp_edge_index(
    edge_index: torch.LongTensor,
    val_pos: np.ndarray,
    test_pos: np.ndarray
) -> torch.LongTensor:
    """
    Construct message-passing edge index with leakage prevention.
    
    Remove all validation and test positive edges from the full graph.
    Graph is undirected, so remove both (u,v) and (v,u).
    
    Args:
        edge_index: Full graph edge index (2, E)
        val_pos: Validation positive edges (N_val, 2)
        test_pos: Test positive edges (N_test, 2)
        
    Returns:
        edge_index_mp: Filtered edge index
    """
    # Create set of edges to remove
    remove_set = set()
    
    for edge in val_pos:
        u, v = int(edge[0]), int(edge[1])
        remove_set.add((min(u, v), max(u, v)))
    
    for edge in test_pos:
        u, v = int(edge[0]), int(edge[1])
        remove_set.add((min(u, v), max(u, v)))
    
    # Filter edge_index
    keep_mask = []
    
    for i in range(edge_index.shape[1]):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        edge = (min(u, v), max(u, v))
        
        if edge not in remove_set:
            keep_mask.append(True)
        else:
            keep_mask.append(False)
    
    keep_mask = torch.tensor(keep_mask, dtype=torch.bool)
    edge_index_mp = edge_index[:, keep_mask]
    
    print(f"  Full graph edges: {edge_index.shape[1]}")
    print(f"  Removed (val+test): {(~keep_mask).sum().item()}")
    print(f"  Message-passing edges: {edge_index_mp.shape[1]}")
    
    return edge_index_mp


def _check_no_leakage(
    edge_index_mp: torch.LongTensor,
    val_pos: np.ndarray,
    test_pos: np.ndarray,
    label: str
):
    """
    Assert that no validation or test positive edges appear in edge_index_mp.
    
    Raises AssertionError if leakage detected.
    """
    # Create set of MP edges
    mp_edge_set = set()
    for i in range(edge_index_mp.shape[1]):
        u, v = edge_index_mp[0, i].item(), edge_index_mp[1, i].item()
        mp_edge_set.add((min(u, v), max(u, v)))
    
    # Check val edges
    for edge in val_pos:
        u, v = int(edge[0]), int(edge[1])
        edge_tuple = (min(u, v), max(u, v))
        assert edge_tuple not in mp_edge_set, f"LEAKAGE: val edge {edge_tuple} found in MP graph!"
    
    # Check test edges
    for edge in test_pos:
        u, v = int(edge[0]), int(edge[1])
        edge_tuple = (min(u, v), max(u, v))
        assert edge_tuple not in mp_edge_set, f"LEAKAGE: test edge {edge_tuple} found in MP graph!"
    
    print(f"✓ Leakage check PASS for {label}")


def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test splits")
    parser.add_argument('--pkl', type=str, required=True, help='Path to graph pickle file')
    parser.add_argument('--mode', type=str, required=True, choices=['single', 'kfold'],
                        help='Split mode: single or kfold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--k', type=int, default=5, help='Number of folds (for kfold mode)')
    parser.add_argument('--out_dir', type=str, default='splits', help='Output directory')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing splits')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        generate_single_split(
            pkl_path=args.pkl,
            out_dir=args.out_dir,
            seed=args.seed,
            overwrite=args.overwrite
        )
    elif args.mode == 'kfold':
        generate_kfold_splits(
            pkl_path=args.pkl,
            out_dir=args.out_dir,
            seed=args.seed,
            k=args.k,
            overwrite=args.overwrite
        )


if __name__ == '__main__':
    main()
