"""
GCN (Graph Convolutional Network) model for link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Standard GCN encoder for node embeddings.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        num_layers: Number of GCN layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Last layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            # Single layer: direct mapping
            self.convs[0] = GCNConv(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        Forward pass to compute node embeddings.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge index for message passing (2, num_edges)
            
        Returns:
            Node embeddings (num_nodes, out_channels)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # Apply ReLU and dropout for all layers except the last
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def decode(self, z, edge_label_index):
        """
        Decode edge scores from node embeddings.
        
        Uses dot product: score = z[u] · z[v]
        
        Args:
            z: Node embeddings (num_nodes, out_channels)
            edge_label_index: Edge index to predict (2, num_edges)
            
        Returns:
            Edge scores (num_edges,)
        """
        src = edge_label_index[0]
        dst = edge_label_index[1]
        
        # Dot product
        scores = (z[src] * z[dst]).sum(dim=-1)
        
        return scores
