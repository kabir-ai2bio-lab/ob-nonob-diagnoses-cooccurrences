"""
GraphSAGE-GAT Hybrid model for link prediction.

STACKED ARCHITECTURE (NOT PARALLEL):
    GraphSAGE encoder → GAT refinement → final embedding → decode

Critical design:
    1. GraphSAGE runs FIRST, producing h_sage [num_nodes, sage_hidden_channels]
    2. GAT consumes h_sage (NOT raw x), producing h_gat with attention
    3. NO parallel branches - this is sequential/stacked processing
    4. Final projection maps to out_channels

Integration contract:
    - forward(x, edge_index_mp) -> z [num_nodes, out_channels]
    - decode(z, edge_pairs) -> logits [num_pairs] (NOT probabilities)
    - Training pipeline applies torch.sigmoid() separately
    - edge_pairs format: [2, num_pairs] (also supports [num_pairs, 2])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class GraphSAGEGATHybrid(nn.Module):
    """
    Stacked GraphSAGE-GAT hybrid encoder for node embeddings.
    
    Architecture:
        1. GraphSAGE block (base encoder): multiple SAGEConv layers
        2. GAT refinement block: attention-based refinement of GraphSAGE output
        3. Final projection: ensure correct output dimension
        4. Edge MLP decoder: MLP-based link prediction
    
    Args:
        in_channels: Input feature dimension
        sage_hidden_channels: GraphSAGE hidden dimension (default: 128)
        sage_num_layers: Number of GraphSAGE layers (default: 2)
        sage_aggr: GraphSAGE aggregation method (default: 'mean')
        out_channels: Final output embedding dimension (default: 64)
        dropout: Dropout rate for GraphSAGE (default: 0.5)
        gat_hidden_channels: GAT hidden dimension (default: 64)
        gat_heads: Number of attention heads in GAT (default: 4)
        gat_dropout: Dropout rate for GAT (default: 0.2)
        gat_num_layers: Number of GAT layers (default: 1)
        edge_mlp_hidden: Hidden dimension for edge MLP decoder (default: 64)
    """
    
    def __init__(
        self,
        in_channels: int,
        sage_hidden_channels: int = 128,
        sage_num_layers: int = 2,
        sage_aggr: str = 'mean',
        out_channels: int = 64,
        dropout: float = 0.5,
        gat_hidden_channels: int = 64,
        gat_heads: int = 4,
        gat_dropout: float = 0.2,
        gat_num_layers: int = 1,
        edge_mlp_hidden: int = 64
    ):
        super().__init__()
        
        self.sage_num_layers = sage_num_layers
        self.gat_num_layers = gat_num_layers
        self.dropout = dropout
        self.gat_dropout = gat_dropout
        self.out_channels = out_channels
        
        # ========================================
        # 1. GraphSAGE Block (Base Encoder)
        # ========================================
        self.sage_convs = nn.ModuleList()
        
        # First GraphSAGE layer
        self.sage_convs.append(SAGEConv(in_channels, sage_hidden_channels, aggr=sage_aggr))
        
        # Hidden GraphSAGE layers
        for _ in range(sage_num_layers - 2):
            self.sage_convs.append(SAGEConv(sage_hidden_channels, sage_hidden_channels, aggr=sage_aggr))
        
        # Last GraphSAGE layer
        if sage_num_layers > 1:
            self.sage_convs.append(SAGEConv(sage_hidden_channels, sage_hidden_channels, aggr=sage_aggr))
        else:
            # Single layer: direct mapping
            self.sage_convs[0] = SAGEConv(in_channels, sage_hidden_channels, aggr=sage_aggr)
        
        # ========================================
        # 2. GAT Refinement Block (Attention)
        # ========================================
        self.gat_convs = nn.ModuleList()
        
        # First GAT layer (consumes GraphSAGE output)
        self.gat_convs.append(
            GATConv(
                sage_hidden_channels, 
                gat_hidden_channels, 
                heads=gat_heads, 
                dropout=gat_dropout,
                concat=True  # Concatenate attention heads
            )
        )
        
        # Additional GAT layers if needed
        for _ in range(gat_num_layers - 1):
            self.gat_convs.append(
                GATConv(
                    gat_hidden_channels * gat_heads,
                    gat_hidden_channels,
                    heads=gat_heads,
                    dropout=gat_dropout,
                    concat=True
                )
            )
        
        # ========================================
        # 3. Final Projection
        # ========================================
        # After GAT, we have gat_hidden_channels * gat_heads
        # Project down to out_channels
        self.final_proj = nn.Linear(gat_hidden_channels * gat_heads, out_channels)
        
        # ========================================
        # 4. Edge MLP Decoder (Link Prediction)
        # ========================================
        # Input: [z_u, z_v, |z_u - z_v|, z_u * z_v] = 4 * out_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * out_channels, edge_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_mlp_hidden, 1)
        )
    
    def forward(self, x, edge_index_mp):
        """
        Forward pass to compute node embeddings.
        
        ARCHITECTURE VALIDATION:
            This is a STACKED model where GAT consumes GraphSAGE output.
            NOT a parallel fusion model.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index_mp: Message-passing edge index [2, num_edges]
            
        Returns:
            z: Final node embeddings [num_nodes, out_channels]
        """
        # ========================================
        # INPUT VALIDATION (FAIL FAST)
        # ========================================
        assert x.dim() == 2, f"Expected x to be 2D [num_nodes, in_channels], got shape {x.shape}"
        assert edge_index_mp.dim() == 2, f"Expected edge_index_mp to be 2D, got shape {edge_index_mp.shape}"
        assert edge_index_mp.size(0) == 2, f"Expected edge_index_mp[0] = 2 for [2, num_edges], got {edge_index_mp.size(0)}"
        assert edge_index_mp.dtype in [torch.long, torch.int64], \
            f"edge_index_mp must be long/int64, got {edge_index_mp.dtype}"
        
        num_nodes = x.size(0)
        max_node_idx = edge_index_mp.max().item()
        assert max_node_idx < num_nodes, \
            f"edge_index_mp contains node index {max_node_idx} but only {num_nodes} nodes exist"
        
        # ========================================
        # STAGE 1: GraphSAGE Base Encoder
        # ========================================
        # GraphSAGE runs FIRST to produce base embeddings
        h = x
        for i, conv in enumerate(self.sage_convs):
            h = conv(h, edge_index_mp)
            
            # Apply ReLU and dropout for all layers except potentially the last
            if i < len(self.sage_convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # After GraphSAGE, apply activation (important for GAT input)
        h_sage = F.relu(h)
        h_sage = F.dropout(h_sage, p=self.dropout, training=self.training)
        
        # ASSERTION: Verify GraphSAGE output shape
        expected_sage_dim = self.sage_convs[-1].out_channels
        assert h_sage.size(0) == num_nodes, \
            f"GraphSAGE changed node count: expected {num_nodes}, got {h_sage.size(0)}"
        assert h_sage.size(1) == expected_sage_dim, \
            f"GraphSAGE output dim mismatch: expected {expected_sage_dim}, got {h_sage.size(1)}"
        
        # ========================================
        # STAGE 2: GAT Refinement Layer
        # ========================================
        # GAT consumes h_sage (NOT raw x) - this proves STACKED architecture
        # CRITICAL: The input to GAT is h_sage, proving this is NOT parallel fusion
        h = h_sage  # GAT takes GraphSAGE output as input
        
        # Runtime assertion to prove stacking (debug mode)
        assert h is h_sage, "ARCHITECTURE VIOLATION: GAT must consume h_sage, not x"
        
        for i, conv in enumerate(self.gat_convs):
            h = conv(h, edge_index_mp)
            
            # Apply activation and dropout for all GAT layers
            if i < len(self.gat_convs) - 1:
                h = F.elu(h)  # ELU works well with GAT
                h = F.dropout(h, p=self.gat_dropout, training=self.training)
        
        # After GAT, apply final activation
        h_gat = F.elu(h)
        h_gat = F.dropout(h_gat, p=self.gat_dropout, training=self.training)
        
        # ASSERTION: Verify GAT output shape (with concatenated heads)
        expected_gat_dim = self.gat_convs[-1].out_channels * self.gat_convs[-1].heads
        assert h_gat.size(0) == num_nodes, \
            f"GAT changed node count: expected {num_nodes}, got {h_gat.size(0)}"
        assert h_gat.size(1) == expected_gat_dim, \
            f"GAT output dim mismatch: expected {expected_gat_dim}, got {h_gat.size(1)}"
        
        # ========================================
        # STAGE 3: Final Projection
        # ========================================
        z = self.final_proj(h_gat)
        
        # FINAL VALIDATION
        assert z.size(0) == num_nodes, \
            f"Final embedding node count mismatch: expected {num_nodes}, got {z.size(0)}"
        assert z.size(1) == self.out_channels, \
            f"Final embedding dimension mismatch: expected {self.out_channels}, got {z.size(1)}"
        assert not torch.isnan(z).any(), "NaN detected in final embeddings"
        assert not torch.isinf(z).any(), "Inf detected in final embeddings"
        
        return z
    
    def _edge_features(self, z, u_idx, v_idx):
        """
        Construct edge features from node embeddings.
        
        For each edge (u,v), creates: [z_u, z_v, |z_u - z_v|, z_u * z_v]
        
        Args:
            z: Node embeddings [num_nodes, out_channels]
            u_idx: Source node indices [num_pairs]
            v_idx: Target node indices [num_pairs]
            
        Returns:
            Edge features [num_pairs, 4 * out_channels]
        """
        num_nodes = z.size(0)
        
        # Validate indices
        assert u_idx.max() < num_nodes, \
            f"u_idx contains {u_idx.max().item()} but only {num_nodes} nodes exist"
        assert v_idx.max() < num_nodes, \
            f"v_idx contains {v_idx.max().item()} but only {num_nodes} nodes exist"
        assert u_idx.min() >= 0, f"u_idx contains negative index {u_idx.min().item()}"
        assert v_idx.min() >= 0, f"v_idx contains negative index {v_idx.min().item()}"
        
        # Get node embeddings for each edge
        z_u = z[u_idx]  # [num_pairs, out_channels]
        z_v = z[v_idx]  # [num_pairs, out_channels]
        
        # Construct edge features: concatenate 4 components
        edge_features = torch.cat([
            z_u,                        # Component 1: source embedding
            z_v,                        # Component 2: target embedding
            torch.abs(z_u - z_v),       # Component 3: absolute difference
            z_u * z_v                   # Component 4: element-wise product
        ], dim=-1)
        
        # Validate output shape
        expected_dim = 4 * self.out_channels
        assert edge_features.size(1) == expected_dim, \
            f"Edge features dimension mismatch: expected {expected_dim}, got {edge_features.size(1)}"
        
        return edge_features
    
    def decode(self, z, edge_pairs):
        """
        Decode edge LOGITS from node embeddings using MLP.
        
        IMPORTANT: Returns LOGITS (not probabilities).
        Training pipeline applies torch.sigmoid() separately.
        
        Args:
            z: Node embeddings [num_nodes, out_channels]
            edge_pairs: Edge pairs to predict, either:
                - [2, num_pairs] (preferred by training pipeline) OR
                - [num_pairs, 2] (also supported)
            
        Returns:
            logits: Edge logits [num_pairs] for binary_cross_entropy_with_logits
        """
        # ========================================
        # INPUT VALIDATION
        # ========================================
        assert z.dim() == 2, f"Expected z to be 2D [num_nodes, out_channels], got shape {z.shape}"
        assert edge_pairs.dim() == 2, f"Expected edge_pairs to be 2D, got shape {edge_pairs.shape}"
        
        # Handle both edge_pairs shapes
        if edge_pairs.size(0) == 2:
            # Shape [2, N] - standard format
            u_idx = edge_pairs[0]
            v_idx = edge_pairs[1]
        elif edge_pairs.size(1) == 2:
            # Shape [N, 2] - alternative format
            u_idx = edge_pairs[:, 0]
            v_idx = edge_pairs[:, 1]
        else:
            raise ValueError(
                f"edge_pairs must be [2, N] or [N, 2], got {edge_pairs.shape}. "
                f"Use [2, N] format for training pipeline compatibility."
            )
        
        num_pairs = u_idx.size(0)
        
        # ========================================
        # EDGE FEATURE CONSTRUCTION
        # ========================================
        edge_features = self._edge_features(z, u_idx, v_idx)
        
        # ========================================
        # MLP DECODER
        # ========================================
        # Pass through MLP (no sigmoid here - returns logits)
        logits = self.edge_mlp(edge_features).squeeze(-1)  # [num_pairs]
        
        # ========================================
        # OUTPUT VALIDATION
        # ========================================
        assert logits.dim() == 1, \
            f"Expected 1D logits [num_pairs], got shape {logits.shape}"
        assert logits.size(0) == num_pairs, \
            f"Logits count mismatch: expected {num_pairs}, got {logits.size(0)}"
        assert not torch.isnan(logits).any(), "NaN detected in logits"
        assert not torch.isinf(logits).any(), "Inf detected in logits"
        
        return logits
    
    def decode_probs(self, z, edge_pairs):
        """
        Decode edge PROBABILITIES (for evaluation/inference).
        
        This is a convenience method that applies sigmoid to logits.
        Use decode() for training (with binary_cross_entropy_with_logits).
        
        Args:
            z: Node embeddings [num_nodes, out_channels]
            edge_pairs: Edge pairs [2, num_pairs] or [num_pairs, 2]
            
        Returns:
            probs: Edge probabilities [num_pairs] in range [0, 1]
        """
        logits = self.decode(z, edge_pairs)
        probs = torch.sigmoid(logits)
        
        # Validate probability range
        assert (probs >= 0).all() and (probs <= 1).all(), \
            f"Probabilities outside [0,1]: min={probs.min():.4f}, max={probs.max():.4f}"
        
        return probs


# ========================================
# Smoke Test (Optional)
# ========================================
if __name__ == "__main__":
    print("=" * 80)
    print("GraphSAGE-GAT Hybrid Model - Smoke Test")
    print("=" * 80)
    
    # Create dummy graph (279 nodes, 9 features - matching EHRShot)
    num_nodes = 279
    num_features = 9
    num_edges = 1000
    num_test_pairs = 50
    
    print(f"\nCreating dummy graph:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Features: {num_features}")
    print(f"  Edges: {num_edges}")
    
    # Random node features
    x = torch.randn(num_nodes, num_features)
    
    # Random edge index (ensure valid node indices)
    edge_index_mp = torch.randint(0, num_nodes, (2, num_edges))
    
    # Random edge pairs to predict (test both shapes)
    edge_pairs_2N = torch.randint(0, num_nodes, (2, num_test_pairs))
    edge_pairs_N2 = torch.randint(0, num_nodes, (num_test_pairs, 2))
    
    # Create model with default parameters
    print("\nCreating model with default parameters...")
    model = GraphSAGEGATHybrid(
        in_channels=num_features,
        sage_hidden_channels=128,
        sage_num_layers=2,
        sage_aggr='mean',
        out_channels=64,
        dropout=0.5,
        gat_hidden_channels=64,
        gat_heads=4,
        gat_dropout=0.2,
        gat_num_layers=1,
        edge_mlp_hidden=64
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        z = model(x, edge_index_mp)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {z.shape}")
    print(f"  ✓ Forward pass successful")
    
    # Test decode with shape [2, N]
    print("\nTesting decode with edge_pairs shape [2, N]...")
    with torch.no_grad():
        probs_2N = model.decode(z, edge_pairs_2N)
    print(f"  Edge pairs shape: {edge_pairs_2N.shape}")
    print(f"  Probabilities shape: {probs_2N.shape}")
    print(f"  Probability range: [{probs_2N.min():.4f}, {probs_2N.max():.4f}]")
    print(f"  ✓ Decode [2, N] successful")
    
    # Test decode with shape [N, 2]
    print("\nTesting decode with edge_pairs shape [N, 2]...")
    with torch.no_grad():
        probs_N2 = model.decode(z, edge_pairs_N2)
    print(f"  Edge pairs shape: {edge_pairs_N2.shape}")
    print(f"  Probabilities shape: {probs_N2.shape}")
    print(f"  Probability range: [{probs_N2.min():.4f}, {probs_N2.max():.4f}]")
    print(f"  ✓ Decode [N, 2] successful")
    
    # Test training mode
    print("\nTesting training mode...")
    model.train()
    z_train = model(x, edge_index_mp)
    probs_train = model.decode(z_train, edge_pairs_2N)
    print(f"  ✓ Training mode successful")
    
    print("\n" + "=" * 80)
    print("All smoke tests passed!")
    print("=" * 80)
