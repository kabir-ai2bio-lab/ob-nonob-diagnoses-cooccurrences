"""
GraphSAGE model for link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """
    GraphSAGE encoder for node embeddings.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        num_layers: Number of SAGE layers
        dropout: Dropout probability
        aggr: Aggregation method ('mean', 'max', 'lstm')
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        aggr: str = 'mean'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        
        # Last layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
        else:
            # Single layer
            self.convs[0] = SAGEConv(in_channels, out_channels, aggr=aggr)
    
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
