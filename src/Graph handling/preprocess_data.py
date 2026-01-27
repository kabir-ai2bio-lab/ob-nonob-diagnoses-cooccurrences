#!/usr/bin/env python3
"""
preprocess_data.py
Build and save the pregnancy diagnosis co-occurrence graph for GNN models.
This should be run once to prepare data for GCN, GAT, GraphSAGE, etc.
"""

import os
import sys
import pickle
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from project2_gcn import build_pregnancy_graph_and_features

def main():
    print("="*70)
    print("EHRShot GNN Data Preprocessing")
    print("="*70)
    print()
    
    # Build the pregnancy graph and features
    print("Building pregnancy diagnosis co-occurrence graph...")
    nodes, X, edge_pairs, ob_mask = build_pregnancy_graph_and_features()
    
    # Create data directory
    data_dir = "preprocessed_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the data
    output_file = os.path.join(data_dir, "pregnancy_graph_data.pkl")
    
    data = {
        'nodes': nodes,
        'X': X,
        'edge_pairs': edge_pairs,
        'ob_mask': ob_mask,
        'num_nodes': len(nodes),
        'num_features': X.shape[1],
        'num_edges': edge_pairs.shape[0]
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print()
    print("="*70)
    print("Data preprocessing complete!")
    print("="*70)
    print(f"Saved to: {output_file}")
    print()
    print("Data summary:")
    print(f"  Number of nodes: {data['num_nodes']}")
    print(f"  Number of features: {data['num_features']}")
    print(f"  Number of edges: {data['num_edges']}")
    print(f"  Obstetric nodes: {ob_mask.sum()}")
    print(f"  Non-obstetric nodes: {(~ob_mask).sum()}")
    print()
    print("You can now run GNN models using this preprocessed data:")
    print("  sbatch run_gcn.sh")
    print("  sbatch run_gat.sh")
    print("  sbatch run_graphsage.sh")
    print("="*70)

if __name__ == "__main__":
    main()
