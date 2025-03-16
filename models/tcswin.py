import torch
import torch.nn as nn
import torch.nn.functional as F

class TCSwinBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Layer norms
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        # Multi-head attention components
        self.qkv = nn.Linear(in_dim, out_dim * 3)
        self.proj = nn.Linear(out_dim, out_dim)

        # Dimension matching for residual
        self.dim_match = None if in_dim == out_dim else nn.Linear(in_dim, out_dim)

        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Linear(out_dim * 4, out_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, N, C]
        Returns:
            x: Output tensor [B, N, out_dim]
        """
        B, N, C = x.shape
        assert C == self.in_dim, f"Input dimension {C} doesn't match expected {self.in_dim}"

        try:
            # Self attention
            shortcut = x
            x = self.norm1(x)

            # QKV transformation
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
            q, k, v = qkv.unbind(0)  # Each: [B, H, N, D]

            # Attention computation
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, self.out_dim)
            x = self.proj(x)

            # Handle residual with dimension matching
            if self.dim_match is not None:
                shortcut = self.dim_match(shortcut)
            x = shortcut + x

            # MLP block with residual
            shortcut = x
            x = self.norm2(x)
            x = shortcut + self.mlp(x)

            return x

        except RuntimeError as e:
            print(f"Error in TCSwinBlock forward pass: {str(e)}")
            print(f"Shapes: x={x.shape}, in_dim={self.in_dim}, out_dim={self.out_dim}")
            raise

class TCSwin(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.patches_resolution = img_size // patch_size

        # Initial patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # Calculate feature dimensions for each stage
        self.num_stages = len(depths)
        self.dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]

        # Create stages
        self.stages = nn.ModuleList()
        self.boundary_convs = nn.ModuleList()

        for i in range(self.num_stages):
            # Create blocks for current stage
            stage_blocks = []
            for j in range(depths[i]):
                # Handle dimension change at start of stage
                in_dim = self.dims[i-1] if i > 0 and j == 0 else self.dims[i]
                out_dim = self.dims[i]

                block = TCSwinBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_heads=num_heads
                )
                stage_blocks.append(block)

            self.stages.append(nn.Sequential(*stage_blocks))

            # Boundary attention projection back to embed_dim
            self.boundary_convs.append(
                nn.Sequential(
                    nn.Linear(self.dims[i], embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU()
                )
            )

    def forward(self, x):
        """
        Args:
            x: Input image [B, C, H, W]
        Returns:
            boundary_maps: List of boundary attention maps [B, embed_dim, H_i, W_i]
        """
        try:
            B = x.shape[0]
            boundary_maps = []

            # Initial patch embedding
            x = self.patch_embed(x)  # [B, embed_dim, H/P, W/P]
            x = x.flatten(2).transpose(1, 2)  # [B, (H/P)*(W/P), embed_dim]

            # Track spatial dimensions
            curr_resolution = self.patches_resolution
            H = W = curr_resolution

            # Process through stages
            for i, (stage, boundary_conv) in enumerate(zip(self.stages, self.boundary_convs)):
                # Process through transformer blocks
                x = stage(x)  # [B, num_patches, dims[i]]

                # Generate boundary attention map
                boundary_feat = boundary_conv(x)  # [B, num_patches, embed_dim]
                boundary_map = boundary_feat.transpose(1, 2).view(B, -1, H, W)
                boundary_maps.append(boundary_map)

                # Prepare for next stage (except last)
                if i < len(self.stages) - 1:
                    # Update resolution
                    curr_resolution = curr_resolution // 2
                    H, W = curr_resolution, curr_resolution

                    # Reshape for spatial operations
                    x = x.view(B, H * 2, W * 2, -1)  # [B, H*2, W*2, C]
                    x = x.permute(0, 3, 1, 2)  # [B, C, H*2, W*2]
                    x = F.avg_pool2d(x, kernel_size=2)  # [B, C, H, W]
                    x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

            return boundary_maps

        except RuntimeError as e:
            print(f"Error in TCSwin forward pass: {str(e)}")
            print(f"Input shape: {x.shape}")
            print(f"Current stage: {i}, resolution: {curr_resolution}")
            raise