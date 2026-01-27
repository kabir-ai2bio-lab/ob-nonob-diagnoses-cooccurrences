"""
Hybrid GCN+GAT model with LayerNorm and fusion mechanism.

CORRECTED ARCHITECTURE:
- Both GCN and GAT branches output same embedding dimension
- LayerNorm applied to each branch before fusion
- Fusion options: projected_concat or gated_sum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class HybridGCNGAT(nn.Module):
    """
    Hybrid GCN+GAT encoder with proper fusion.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension (same for both branches)
        num_layers: Number of layers per branch
        gcn_dropout: Dropout for GCN branch
        gat_dropout: Dropout for GAT branch
        gat_heads: Number of attention heads for GAT
        fusion_method: 'projected_concat' or 'gated_sum'
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        gcn_dropout: float = 0.0,
        gat_dropout: float = 0.0,
        gat_heads: int = 4,
        fusion_method: str = 'projected_concat'
    ):
        super().__init__()
        
        assert fusion_method in ['projected_concat', 'gated_sum'], \
            f"fusion_method must be 'projected_concat' or 'gated_sum', got {fusion_method}"
        
        self.num_layers = num_layers
        self.gcn_dropout = gcn_dropout
        self.gat_dropout = gat_dropout
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
        
        # GAT branch
        self.gat_convs = nn.ModuleList()
        self.gat_convs.append(
            GATConv(in_channels, hidden_channels, heads=gat_heads, dropout=gat_dropout, concat=True)
        )
        gat_hidden_dim = hidden_channels * gat_heads
        for _ in range(num_layers - 2):
            self.gat_convs.append(
                GATConv(gat_hidden_dim, hidden_channels, heads=gat_heads, dropout=gat_dropout, concat=True)
            )
        if num_layers > 1:
            self.gat_convs.append(
                GATConv(gat_hidden_dim, out_channels, heads=gat_heads, dropout=gat_dropout, concat=False)
            )
        else:
            self.gat_convs[0] = GATConv(in_channels, out_channels, heads=gat_heads, dropout=gat_dropout, concat=False)
        
        # LayerNorm for both branches (output dimension)
        self.gcn_norm = nn.LayerNorm(out_channels)
        self.gat_norm = nn.LayerNorm(out_channels)
        
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
        
        # GAT branch
        x_gat = x
        for i, conv in enumerate(self.gat_convs):
            x_gat = conv(x_gat, edge_index)
            if i < len(self.gat_convs) - 1:
                x_gat = F.elu(x_gat)
                x_gat = F.dropout(x_gat, p=self.gat_dropout, training=self.training)
        
        # Apply LayerNorm to both branches
        x_gcn = self.gcn_norm(x_gcn)
        x_gat = self.gat_norm(x_gat)
        
        # Fusion
        if self.fusion_method == 'projected_concat':
            # Concatenate and project
            x_cat = torch.cat([x_gcn, x_gat], dim=-1)
            x_fused = self.fusion_proj(x_cat)
        elif self.fusion_method == 'gated_sum':
            # Gated weighted sum
            x_cat = torch.cat([x_gcn, x_gat], dim=-1)
            gate = torch.sigmoid(self.gate_fc(x_cat))  # (num_nodes, 1)
            x_fused = gate * x_gcn + (1 - gate) * x_gat
        
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
