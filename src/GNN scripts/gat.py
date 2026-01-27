"""
GAT (Graph Attention Network) model for link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    GAT encoder for node embeddings with multi-head attention.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension (per head)
        out_channels: Output embedding dimension
        num_layers: Number of GAT layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        concat_heads: If True, concatenate heads; if False, average them
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        concat_heads: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=num_heads,
                dropout=dropout,
                concat=concat_heads
            )
        )
        
        # Determine input dimension for subsequent layers
        first_out_dim = hidden_channels * num_heads if concat_heads else hidden_channels
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    first_out_dim,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=concat_heads
                )
            )
        
        # Last layer: average heads to get fixed output dimension
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    first_out_dim,
                    out_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False  # Average heads in final layer
                )
            )
        else:
            # Single layer case
            self.convs[0] = GATConv(
                in_channels,
                out_channels,
                heads=num_heads,
                dropout=dropout,
                concat=False
            )
    
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
            
            # Apply ELU and dropout for all layers except the last
            if i < len(self.convs) - 1:
                x = F.elu(x)
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
