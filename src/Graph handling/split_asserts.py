"""
Training-time safety assertions for split integrity.

Add these checks at the start of training scripts to ensure
splits are valid before wasting GPU time.
"""

import numpy as np
import torch
from typing import Tuple, Set


def canonicalize_edge(u: int, v: int) -> Tuple[int, int]:
    """Return (min, max) canonical form."""
    return (min(u, v), max(u, v))


def edges_to_set(edges: np.ndarray) -> Set[Tuple[int, int]]:
    """Convert edge array (N, 2) to set of canonical tuples."""
    edge_set = set()
    for i in range(len(edges)):
        u, v = int(edges[i, 0]), int(edges[i, 1])
        edge_set.add(canonicalize_edge(u, v))
    return edge_set


def assert_split_integrity(
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    test_pos: np.ndarray,
    train_neg: np.ndarray,
    val_neg: np.ndarray,
    test_neg: np.ndarray,
    edge_index_mp: torch.LongTensor,
    edge_index_full: torch.LongTensor,
    fold_id: int = None,
    is_obstetric: torch.Tensor = None
):
    """
    Fast runtime assertion of split integrity.
    
    Raises AssertionError if any check fails.
    Should be called BEFORE training begins.
    """
    fold_label = f"fold {fold_id}" if fold_id is not None else "split"
    
    # Convert to sets
    train_pos_set = edges_to_set(train_pos)
    val_pos_set = edges_to_set(val_pos)
    test_pos_set = edges_to_set(test_pos)
    train_neg_set = edges_to_set(train_neg)
    val_neg_set = edges_to_set(val_neg)
    test_neg_set = edges_to_set(test_neg)
    
    # 1. Disjointness
    assert len(train_pos_set & val_pos_set) == 0, f"{fold_label}: train_pos and val_pos overlap!"
    assert len(train_pos_set & test_pos_set) == 0, f"{fold_label}: train_pos and test_pos overlap!"
    assert len(val_pos_set & test_pos_set) == 0, f"{fold_label}: val_pos and test_pos overlap!"
    assert len(train_neg_set & val_neg_set) == 0, f"{fold_label}: train_neg and val_neg overlap!"
    assert len(train_neg_set & test_neg_set) == 0, f"{fold_label}: train_neg and test_neg overlap!"
    assert len(val_neg_set & test_neg_set) == 0, f"{fold_label}: val_neg and test_neg overlap!"
    
    all_pos = train_pos_set | val_pos_set | test_pos_set
    all_neg = train_neg_set | val_neg_set | test_neg_set
    assert len(all_pos & all_neg) == 0, f"{fold_label}: positives and negatives overlap!"
    
    # 2. Balance
    assert len(train_neg) == len(train_pos), \
        f"{fold_label}: train negatives ({len(train_neg)}) != positives ({len(train_pos)})"
    assert len(val_neg) == len(val_pos), \
        f"{fold_label}: val negatives ({len(val_neg)}) != positives ({len(val_pos)})"
    assert len(test_neg) == len(test_pos), \
        f"{fold_label}: test negatives ({len(test_neg)}) != positives ({len(test_pos)})"
    
    # 3. Negatives not in full graph
    full_graph_edges = set()
    for i in range(edge_index_full.shape[1]):
        u, v = edge_index_full[0, i].item(), edge_index_full[1, i].item()
        full_graph_edges.add(canonicalize_edge(u, v))
    
    neg_in_graph = all_neg & full_graph_edges
    assert len(neg_in_graph) == 0, \
        f"{fold_label}: {len(neg_in_graph)} negatives exist in graph!"
    
    # 4. Leakage check
    mp_edge_set = set()
    for i in range(edge_index_mp.shape[1]):
        u, v = edge_index_mp[0, i].item(), edge_index_mp[1, i].item()
        mp_edge_set.add(canonicalize_edge(u, v))
    
    val_in_mp = val_pos_set & mp_edge_set
    test_in_mp = test_pos_set & mp_edge_set
    
    assert len(val_in_mp) == 0, \
        f"{fold_label}: {len(val_in_mp)} val edges leaked into edge_index_mp!"
    assert len(test_in_mp) == 0, \
        f"{fold_label}: {len(test_in_mp)} test edges leaked into edge_index_mp!"
    
    # 5. Cross-community verification (if is_obstetric provided)
    if is_obstetric is not None:
        def verify_cross_community(edges: np.ndarray, label: str):
            for i in range(len(edges)):
                u, v = int(edges[i, 0]), int(edges[i, 1])
                assert is_obstetric[u] != is_obstetric[v], \
                    f"{fold_label}: {label} edge ({u},{v}) is NOT cross-community!"
        
        verify_cross_community(train_pos, "train_pos")
        verify_cross_community(val_pos, "val_pos")
        verify_cross_community(test_pos, "test_pos")
        verify_cross_community(train_neg, "train_neg")
        verify_cross_community(val_neg, "val_neg")
        verify_cross_community(test_neg, "test_neg")
        print(f"✓ Cross-community verification PASS", flush=True)
    
    print(f"✓ Split integrity checks PASS for {fold_label}", flush=True)
