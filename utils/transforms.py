import torch
import torch.nn.functional as F
import numpy as np
import cv2

class RemoteSensingTransforms:
    def __init__(self, img_size=224):
        self.img_size = img_size

    def normalize_image(self, image):
        """
        Normalize image to [-1, 1] range
        Args:
            image: np.ndarray or torch.Tensor in range [0, 255]
        Returns:
            normalized image in range [-1, 1]
        """
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32) / 127.5 - 1.0
        else:
            image = image.float() / 127.5 - 1.0
        return image

    def resize_image(self, image):
        """
        Resize image to target size
        Args:
            image: Input image [H, W, C] or [C, H, W]
        Returns:
            resized image [C, H, W]
        """
        # Handle input format
        if isinstance(image, np.ndarray):
            if image.shape[0] == 3:  # Already in CHW format
                image = image.transpose(1, 2, 0)  # Convert to HWC for cv2

            # Resize
            image = cv2.resize(image, (self.img_size, self.img_size))

            # Convert back to CHW format
            image = image.transpose(2, 0, 1)

        else:  # torch.Tensor
            if image.shape[0] != 3:  # Not in CHW format
                image = image.permute(2, 0, 1)  # Convert to CHW

            # Resize using F.interpolate
            image = F.interpolate(
                image.unsqueeze(0), 
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        return image

    def prepare_nodes(self, nodes):
        """
        Convert node coordinates to normalized format
        Args:
            nodes: Node coordinates [N, 2] in pixel space
        Returns:
            normalized nodes [N, 2] in [0, 1] range
        """
        nodes = nodes.astype(np.float32)
        nodes = nodes / self.img_size  # Normalize to [0, 1]
        return torch.from_numpy(nodes).float()

    def prepare_adjacency(self, adjacency):
        """
        Convert adjacency matrix to tensor
        Args:
            adjacency: Binary adjacency matrix [N, N]
        Returns:
            adjacency matrix as float tensor
        """
        return torch.from_numpy(adjacency).float()

    def __call__(self, image, nodes=None, adjacency=None):
        """
        Apply all transforms
        Args:
            image: Input image [H, W, C] or [C, H, W]
            nodes: Optional node coordinates [N, 2]
            adjacency: Optional adjacency matrix [N, N]
        Returns:
            transformed image [C, H, W], nodes [N, 2], adjacency [N, N]
        """
        # Normalize and resize image
        image = self.normalize_image(image)
        image = self.resize_image(image)

        # Ensure image is a tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Transform nodes and adjacency if provided
        if nodes is not None and adjacency is not None:
            nodes = self.prepare_nodes(nodes)
            adjacency = self.prepare_adjacency(adjacency)
            return image, nodes, adjacency

        return image