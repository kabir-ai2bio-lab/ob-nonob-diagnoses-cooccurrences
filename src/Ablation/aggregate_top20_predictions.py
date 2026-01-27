"""
Aggregate top N predictions across k-fold runs with concept names and community info.
Uses weighted ranking: mean_probability * log(num_folds + 1) to favor edges appearing in multiple folds.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

def load_concept_names(concept_csv_path, node_ids):
    """Load concept names from concept.csv."""
    df = pd.read_csv(concept_csv_path, dtype={'concept_id': str}, low_memory=False)
    df_filtered = df[df['concept_id'].isin(node_ids)]
    concept_map = dict(zip(df_filtered['concept_id'], df_filtered['concept_name']))
    return concept_map

def load_graph_data(pkl_path):
    """Load graph to get node IDs and obstetric community flags."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    node_ids = data.get('nodes', None)
    is_obstetric = data.get('ob_mask', data.get('is_obstetric', data.get('obstetric_flag', data.get('in_obstetric_community', None))))
    
    # Convert to numpy array if needed
    if is_obstetric is not None and not isinstance(is_obstetric, np.ndarray):
        is_obstetric = np.array(is_obstetric)
    
    return node_ids, is_obstetric

def aggregate_topn_predictions(ablation_dir, pkl_path, concept_csv_path, top_n=50):
    """Aggregate top N predictions across folds with weighted ranking.
    
    Ranking mechanism: mean_probability * log(num_folds + 1)
    This favors edges that:
    1. Have high probability
    2. Appear in multiple folds (recurrence bonus)
    """
    
    # Load graph data
    print("Loading graph data...")
    node_ids, is_obstetric = load_graph_data(pkl_path)
    
    # Load concept names
    print("Loading concept names...")
    concept_map = load_concept_names(concept_csv_path, node_ids)
    node_names = [concept_map.get(node_ids[i], node_ids[i]) for i in range(len(node_ids))]
    
    # Aggregate predictions across folds
    print("\nAggregating predictions across folds...")
    
    # Dictionary to store edge predictions: (source, target) -> [probs from each fold]
    test_edges = {}
    
    # Load from each fold
    for fold_idx in range(5):
        fold_file = Path(ablation_dir) / f'fold_{fold_idx}' / 'top_20_predictions.json'
        with open(fold_file, 'r') as f:
            data = json.load(f)
        
        # Only use test predictions
        for pred in data['test']['top_20_positive']:
            edge = (pred['source'], pred['target'])
            if edge not in test_edges:
                test_edges[edge] = []
            test_edges[edge].append(pred['prob'])
    
    print(f"Found {len(test_edges)} unique edges across folds")
    
    # Calculate mean probabilities and ranking score
    edge_means = []
    for (src, tgt), probs in test_edges.items():
        mean_prob = np.mean(probs)
        num_folds = len(probs)
        # Weighted ranking: favors both high probability and recurrence across folds
        ranking_score = mean_prob * np.log(num_folds + 1)
        
        edge_means.append({
            'source_idx': src,
            'target_idx': tgt,
            'source_name': node_names[src],
            'target_name': node_names[tgt],
            'source_obstetric': bool(is_obstetric[src]),
            'target_obstetric': bool(is_obstetric[tgt]),
            'mean_prob': mean_prob,
            'std_prob': np.std(probs),
            'num_folds': num_folds,
            'ranking_score': ranking_score
        })
    
    # Sort by ranking score (weighted by recurrence)
    edge_means_sorted = sorted(edge_means, key=lambda x: x['ranking_score'], reverse=True)
    
    # Top N
    top_n_edges = edge_means_sorted[:top_n]
    
    # Create DataFrame
    df = pd.DataFrame(top_n_edges)
    
    # Add community column
    df['edge_type'] = df.apply(lambda x: 
        'Within Obstetric' if (x['source_obstetric'] and x['target_obstetric']) else
        'Within Non-Obstetric' if (not x['source_obstetric'] and not x['target_obstetric']) else
        'Cross-Community', axis=1)
    
    # Reorder columns
    df = df[['source_name', 'target_name', 'mean_prob', 'std_prob', 'num_folds',
             'ranking_score', 'source_obstetric', 'target_obstetric', 'edge_type',
             'source_idx', 'target_idx']]
    
    return df

def main():
    ablation_dir = 'ABLATIONS/kfold_runs/ablation_none_frac1.00_20251226_125744'
    pkl_path = '../preprocessed_data/pregnancy_graph_data.pkl'
    concept_csv_path = '../data/EHRShot_sampled_2000patients/concept.csv'
    
    # Get top 50 predictions
    top_n = 50
    df = aggregate_topn_predictions(ablation_dir, pkl_path, concept_csv_path, top_n=top_n)
    
    print("\n" + "="*100)
    print(f"TOP {top_n} PREDICTIONS - BASELINE MODEL (none_frac1.00)")
    print("Aggregated across all 5 folds")
    print("Ranking: mean_probability × log(num_folds + 1) - favors high prob + recurrence")
    print("="*100)
    print(df.to_string(index=True))
    
    # Save to CSV
    output_csv = f'ABLATIONS/kfold_analysis_report/top_{top_n}_predictions_baseline.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved to {output_csv}")
    
    # Print summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"Total unique edges across folds: {len(df)}")
    print(f"\nRecurrence statistics:")
    print(df['num_folds'].value_counts().sort_index(ascending=False))
    
    # Print summary by edge type
    print("\n" + "="*100)
    print("BREAKDOWN BY EDGE TYPE")
    print("="*100)
    print(df['edge_type'].value_counts())
    
    # Print ranking explanation
    print("\n" + "="*100)
    print("RANKING EXPLANATION")
    print("="*100)
    print("Ranking Score = mean_probability × log(num_folds + 1)")
    print("This weights edges by both:")
    print("  1. High prediction probability (mean across folds)")
    print("  2. Recurrence across multiple folds (reliability bonus)")
    print(f"\nExample: An edge appearing in all 5 folds gets {np.log(6):.3f}x weight")
    print(f"         An edge appearing in 1 fold gets {np.log(2):.3f}x weight")
    
    # Check if we can separate by community
    print("\n" + "="*100)
    print("COMMUNITY SEPARATION ANALYSIS")
    print("="*100)
    
    obstetric_concepts = df[df['source_obstetric'] | df['target_obstetric']][['source_name', 'target_name', 'mean_prob', 'edge_type']]
    print(f"\nEdges involving at least one obstetric concept: {len(obstetric_concepts)}")
    
    print("\nFor a two-column layout (Obstetric | Non-Obstetric):")
    print("This would work if edges are strictly within-community.")
    print(f"Cross-community edges: {len(df[df['edge_type'] == 'Cross-Community'])}")
    
    if len(df[df['edge_type'] == 'Cross-Community']) > 0:
        print("\n⚠ Note: Cross-community edges exist, so strict separation isn't clean.")
        print("But we can create a view showing which concepts belong to which community.")

if __name__ == '__main__':
    main()
