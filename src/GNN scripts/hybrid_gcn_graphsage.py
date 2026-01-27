"""
Hybrid GCN+GraphSAGE model with LayerNorm and fusion mechanism.

CORRECTED ARCHITECTURE:
- Both GCN and GraphSAGE branches output same embedding dimension
- LayerNorm applied to each branch before fusion
- Fusion options: projected_concat or gated_sum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class HybridGCNGraphSAGE(nn.Module):
    """
    Hybrid GCN+GraphSAGE encoder with proper fusion.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension (same for both branches)
        num_layers: Number of layers per branch
        gcn_dropout: Dropout for GCN branch
        sage_dropout: Dropout for GraphSAGE branch
        sage_aggr: Aggregation method for GraphSAGE ('mean', 'max', 'lstm')
        fusion_method: 'projected_concat' or 'gated_sum'
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        gcn_dropout: float = 0.0,
        sage_dropout: float = 0.0,
        sage_aggr: str = 'mean',
        fusion_method: str = 'projected_concat'
    ):
        super().__init__()
        
        assert fusion_method in ['projected_concat', 'gated_sum'], \
            f"fusion_method must be 'projected_concat' or 'gated_sum', got {fusion_method}"
        
        self.num_layers = num_layers
        self.gcn_dropout = gcn_dropout
        self.sage_dropout = sage_dropout
        self.fusion_method = fusion_method
        
        # GCN branch
        self.gcn_convs = nn.ModuleList()
        self.gcn_convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
        if num_layers > 1:
            self.gcn_convs.append(GCNConv(hidden_channels, out_channels))
        else:
            self.gcn_convs[0] = GCNConv(in_channels, out_channels)
        
        # GraphSAGE branch
        self.sage_convs = nn.ModuleList()
        self.sage_convs.append(SAGEConv(in_channels, hidden_channels, aggr=sage_aggr))
        for _ in range(num_layers - 2):
            self.sage_convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=sage_aggr))
        if num_layers > 1:
            self.sage_convs.append(SAGEConv(hidden_channels, out_channels, aggr=sage_aggr))
        else:
            self.sage_convs[0] = SAGEConv(in_channels, out_channels, aggr=sage_aggr)
        
        # LayerNorm for both branches (output dimension)
        self.gcn_norm = nn.LayerNorm(out_channels)
        self.sage_norm = nn.LayerNorm(out_channels)
        
        # Fusion layers
        if fusion_method == 'projected_concat':
            # Project concatenated embeddings back to out_channels
            self.fusion_proj = nn.Linear(2 * out_channels, out_channels)
        elif fusion_method == 'gated_sum':
            # Learnable gate for weighted sum
            self.gate_fc = nn.Linear(2 * out_channels, 1)
    
    def forward(self, x, edge_index):
        """
        Forward pass through both branches and fusion.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge index for message passing (2, num_edges)
            
        Returns:
            Fused node embeddings (num_nodes, out_channels)
        """
        # GCN branch
        x_gcn = x
        for i, conv in enumerate(self.gcn_convs):
            x_gcn = conv(x_gcn, edge_index)
            if i < len(self.gcn_convs) - 1:
                x_gcn = F.relu(x_gcn)
                x_gcn = F.dropout(x_gcn, p=self.gcn_dropout, training=self.training)
        
        # GraphSAGE branch
        x_sage = x
        for i, conv in enumerate(self.sage_convs):
            x_sage = conv(x_sage, edge_index)
            if i < len(self.sage_convs) - 1:
                x_sage = F.relu(x_sage)
                x_sage = F.dropout(x_sage, p=self.sage_dropout, training=self.training)
        
        # Apply LayerNorm to both branches
        x_gcn = self.gcn_norm(x_gcn)
        x_sage = self.sage_norm(x_sage)
        
        # Fusion
        if self.fusion_method == 'projected_concat':
            # Concatenate and project
            x_cat = torch.cat([x_gcn, x_sage], dim=-1)
            x_fused = self.fusion_proj(x_cat)
        elif self.fusion_method == 'gated_sum':
            # Gated weighted sum
            x_cat = torch.cat([x_gcn, x_sage], dim=-1)
            gate = torch.sigmoid(self.gate_fc(x_cat))  # (num_nodes, 1)
            x_fused = gate * x_gcn + (1 - gate) * x_sage
        
        return x_fused
    
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
