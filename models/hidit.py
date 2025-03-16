import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HiDiTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Layer norms for each component
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # Self-attention components
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Cross-attention components
        self.cross_q = nn.Linear(dim, dim)
        self.cross_kv = nn.Linear(dim, dim * 2)
        self.cross_proj = nn.Linear(dim, dim)

        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, time_emb, condition):
        """
        Args:
            x: Input features [B, N, C]
            time_emb: Time embedding [B, C]
            condition: Boundary condition [B, L, C]
        Returns:
            x: Transformed features [B, N, C]
        """
        B, N, C = x.shape
        device = x.device

        try:
            # Self attention
            shortcut = x
            x = self.norm1(x)

            # QKV computation for self-attention
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
            q, k, v = qkv.unbind(0)  # Each: [B, H, N, D]

            # Self-attention computation
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
            x = self.proj(x)
            x = shortcut + x

            # Cross attention with condition
            shortcut = x
            x = self.norm2(x)

            # Cross-attention computation
            q = self.cross_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.cross_kv(condition).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)  # Each: [B, H, L, D]

            # Cross-attention computation
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, L]
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.cross_proj(x)
            x = shortcut + x

            # MLP with time embedding
            shortcut = x
            x = self.norm3(x)
            time_emb = time_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, C]
            x = shortcut + self.mlp(x + time_emb)

            return x

        except RuntimeError as e:
            print(f"Error in HiDiTBlock forward pass: {str(e)}")
            print(f"Shapes: x={x.shape}, time_emb={time_emb.shape}, condition={condition.shape}")
            raise

class HiDiT(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=192, num_blocks=6, num_heads=8, max_nodes=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        # Input projections
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.condition_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            HiDiTBlock(hidden_dim, num_heads)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, timesteps, condition):
        """
        Args:
            x: Node features [B, max_nodes, input_dim]
            timesteps: Timestep conditioning [B]
            condition: Boundary map features [B, H*W, input_dim]
        Returns:
            x: Updated node features [B, max_nodes, input_dim]
        """
        try:
            B = x.shape[0]
            device = x.device

            # Project inputs to hidden dimension
            x = self.input_proj(x)  # [B, max_nodes, hidden_dim]
            condition = self.condition_proj(condition)  # [B, H*W, hidden_dim]

            # Create time embeddings
            time_emb = self.time_embed(timesteps)  # [B, hidden_dim]

            # Process through transformer blocks
            for block in self.blocks:
                x = block(x, time_emb, condition)

            # Project back to input dimension
            x = self.output_proj(x)  # [B, max_nodes, input_dim]

            return x

        except RuntimeError as e:
            print(f"Error in HiDiT forward pass: {str(e)}")
            print(f"Input shapes:")
            print(f"- x: {x.shape}")
            print(f"- timesteps: {timesteps.shape}")
            print(f"- condition: {condition.shape}")
            raise

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        """
        Args:
            timesteps: [B] tensor of timesteps
        Returns:
            time_emb: [B, dim] tensor of time embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings