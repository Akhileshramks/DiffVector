import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeBiasedAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Core attention components
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Edge bias components
        self.edge_embed = nn.Linear(dim * 2, dim)
        self.edge_norm = nn.LayerNorm(dim)

    def forward(self, x, edge_features):
        """
        Args:
            x: Node features [B, N, D]
            edge_features: Edge features [B, N*N, D*2]
        Returns:
            x: Updated node features [B, N, D]
        """
        B, N, C = x.shape
        device = x.device

        try:
            # Process edge features
            edge_emb = self.edge_embed(edge_features)  # [B, N*N, C]
            edge_emb = self.edge_norm(edge_emb)
            edge_emb = edge_emb.view(B, N, N, C)  # Reshape to attention format

            # Multi-head attention
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
            q, k, v = qkv.unbind(0)  # Each: [B, num_heads, N, head_dim]

            # Compute attention with edge bias
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
            edge_bias = edge_emb.mean(dim=-1).unsqueeze(1)  # [B, 1, N, N]
            attn = attn + edge_bias
            attn = attn.softmax(dim=-1)

            # Combine with values
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            return x

        except RuntimeError as e:
            print(f"Error in EdgeBiasedAttention: {str(e)}")
            print(f"Shapes: x={x.shape}, edge_features={edge_features.shape}")
            raise

class EGDiTBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.edge_attn = EdgeBiasedAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, edge_features):
        """
        Args:
            x: Node features [B, N, D]
            edge_features: Edge features [B, N*N, D*2]
        Returns:
            x: Transformed node features [B, N, D]
        """
        # Edge-biased attention with residual
        shortcut = x
        x = self.norm1(x)
        x = shortcut + self.edge_attn(x, edge_features)

        # MLP with residual
        shortcut = x
        x = self.norm2(x)
        x = shortcut + self.mlp(x)

        return x

class EGDiT(nn.Module):
    def __init__(self, dim=96, num_blocks=6, num_heads=8):
        super().__init__()
        self.dim = dim

        # Edge embedding projection
        self.edge_proj = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 2)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            EGDiTBlock(dim, num_heads) for _ in range(num_blocks)
        ])

        # Final adjacency prediction
        self.adjacency_proj = nn.Linear(dim * 2, 1)

    def forward(self, x, edge_features):
        """
        Args:
            x: Node features [B, N, D]
            edge_features: Edge features [B, N*N, D*2]
        Returns:
            adjacency: Adjacency matrix [B, N, N]
        """
        try:
            B, N, D = x.shape
            assert D == self.dim, f"Input dimension {D} doesn't match model dimension {self.dim}"

            # Process edge features
            edge_features = self.edge_proj(edge_features)  # [B, N*N, D*2]

            # Process through transformer blocks
            for block in self.blocks:
                x = block(x, edge_features)

            # Predict adjacency matrix
            adjacency = self.adjacency_proj(edge_features)  # [B, N*N, 1]
            adjacency = adjacency.view(B, N, N)  # [B, N, N]

            return adjacency

        except RuntimeError as e:
            print(f"Error in EGDiT forward pass: {str(e)}")
            print(f"Shapes: x={x.shape}, edge_features={edge_features.shape}")
            raise

    def get_edge_embeddings(self, node_features):
        """
        Generate edge embeddings from node features
        Args:
            node_features: [B, N, D]
        Returns:
            edge_features: [B, N*N, 2D]
        """
        try:
            B, N, D = node_features.shape

            # Generate all pairs of node features
            node_i = node_features.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
            node_j = node_features.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]

            # Concatenate node pairs for edge features
            edge_features = torch.cat([node_i, node_j], dim=-1)  # [B, N, N, 2D]
            return edge_features.view(B, N * N, D * 2)

        except RuntimeError as e:
            print(f"Error in get_edge_embeddings: {str(e)}")
            print(f"Shape of node_features: {node_features.shape}")
            raise