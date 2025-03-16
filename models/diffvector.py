import torch
import torch.nn as nn
import torch.nn.functional as F
from .hidit import HiDiT
from .egdit import EGDiT
from .tcswin import TCSwin

class DiffVector(nn.Module):
    def __init__(self, img_size=224, in_channels=3, embed_dim=96, num_heads=8, num_blocks=6, max_nodes=32):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.max_nodes = max_nodes

        # Initialize components
        self.tcswin = TCSwin(
            img_size=img_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads
        )

        self.hidit = HiDiT(
            input_dim=embed_dim,
            hidden_dim=embed_dim * 2,
            num_blocks=num_blocks,
            num_heads=num_heads,
            max_nodes=max_nodes
        )

        self.egdit = EGDiT(
            dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads
        )

        # Node decoder for final position prediction
        self.node_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2)  # x,y coordinates
        )

    def forward(self, x, timesteps=None):
        """
        Args:
            x: Input image [B, C, H, W]
            timesteps: Optional timesteps for diffusion [B]
        Returns:
            Dictionary containing:
            - node_coords: Node coordinates [B, max_nodes, 2]
            - adjacency: Adjacency matrix [B, max_nodes, max_nodes]
            - node_features: Node features [B, max_nodes, embed_dim]
            - boundary_maps: List of boundary attention maps
        """
        try:
            B = x.shape[0]
            device = x.device

            # Debug logging
            print(f"Input shape: {x.shape}")

            # Generate boundary maps using TCSwin
            boundary_maps = self.tcswin(x)  # List of [B, embed_dim, H_i, W_i]
            print(f"Generated {len(boundary_maps)} boundary maps")
            for i, bmap in enumerate(boundary_maps):
                print(f"Boundary map {i} shape: {bmap.shape}")

            # Initialize node features with zeros
            node_features = torch.zeros(B, self.max_nodes, self.embed_dim, device=device)
            print(f"Initial node features shape: {node_features.shape}")

            # Set up timesteps if not provided
            if timesteps is None:
                timesteps = torch.zeros(B, device=device)
            print(f"Timesteps shape: {timesteps.shape}")

            # Use the last boundary map as condition for HiDiT
            last_boundary = boundary_maps[-1]  # [B, embed_dim, H, W]
            last_boundary = last_boundary.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
            print(f"Flattened boundary feature shape: {last_boundary.shape}")

            # Generate node features through HiDiT
            node_features = self.hidit(node_features, timesteps, last_boundary)  # [B, max_nodes, embed_dim]
            print(f"HiDiT output shape: {node_features.shape}")

            # Decode node positions
            node_coords = self.node_mlp(node_features)  # [B, max_nodes, 2]
            node_coords = torch.sigmoid(node_coords)  # Normalize to [0, 1]
            print(f"Node coordinates shape: {node_coords.shape}")

            # Generate edge features and predict adjacency
            edge_features = self.egdit.get_edge_embeddings(node_features)  # [B, max_nodes*max_nodes, embed_dim*2]
            print(f"Edge features shape: {edge_features.shape}")
            adjacency = self.egdit(node_features, edge_features)  # [B, max_nodes, max_nodes]
            print(f"Adjacency matrix shape: {adjacency.shape}")

            return {
                'node_coords': node_coords,
                'adjacency': adjacency,
                'node_features': node_features,
                'boundary_maps': boundary_maps
            }

        except RuntimeError as e:
            print(f"Error in DiffVector forward pass: {str(e)}")
            print(f"Input shapes:")
            print(f"- x: {x.shape}")
            print(f"- node_features: {node_features.shape if 'node_features' in locals() else 'not created'}")
            print(f"- last_boundary: {last_boundary.shape if 'last_boundary' in locals() else 'not created'}")
            raise

    def train_step(self, x, timestep, gt_nodes, gt_adjacency, num_nodes=None):
        """
        Single training step with isomorphic training strategy
        Args:
            x: Input image [B, C, H, W]
            timestep: Current timestep [B]
            gt_nodes: Ground truth node coordinates [B, max_nodes, 2]
            gt_adjacency: Ground truth adjacency matrix [B, max_nodes, max_nodes]
            num_nodes: Number of valid nodes per batch [B]
        Returns:
            total_loss: Combined loss value
            outputs: Model outputs dictionary
        """
        try:
            outputs = self.forward(x, timestep)

            # Calculate losses with masking for valid nodes
            if num_nodes is not None:
                # Create mask for valid nodes
                batch_size = gt_nodes.size(0)
                device = gt_nodes.device
                node_mask = torch.zeros((batch_size, self.max_nodes), device=device)
                for i, n in enumerate(num_nodes):
                    node_mask[i, :n] = 1

                # Apply mask to node loss
                node_loss = F.mse_loss(
                    outputs['node_coords'] * node_mask.unsqueeze(-1),
                    gt_nodes * node_mask.unsqueeze(-1)
                )

                # Apply mask to adjacency loss
                adj_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
                adj_loss = F.binary_cross_entropy_with_logits(
                    outputs['adjacency'] * adj_mask,
                    gt_adjacency * adj_mask,
                    reduction='sum'
                ) / (adj_mask.sum() + 1e-6)
            else:
                node_loss = F.mse_loss(outputs['node_coords'], gt_nodes)
                adj_loss = F.binary_cross_entropy_with_logits(outputs['adjacency'], gt_adjacency)

            total_loss = node_loss + adj_loss
            return total_loss, outputs

        except RuntimeError as e:
            print(f"Error in train_step: {str(e)}")
            print(f"Input shapes:")
            print(f"- x: {x.shape}")
            print(f"- timestep: {timestep.shape}")
            print(f"- gt_nodes: {gt_nodes.shape}")
            print(f"- gt_adjacency: {gt_adjacency.shape}")
            if num_nodes is not None:
                print(f"- num_nodes: {num_nodes.shape}")
            raise