import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import json
from pycocotools.coco import COCO
import logging

class RemoteSensingDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', max_nodes=32, img_size=512):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.max_nodes = max_nodes
        self.img_size = img_size  # Added for normalization

        # Setup logging
        self.logger = logging.getLogger(f'RemoteSensingDataset_{split}')
        self.logger.setLevel(logging.INFO)

        # Initialize COCO
        ann_file = os.path.join(data_dir, split, 'annotation.json')
        self.coco = COCO(ann_file)
        self.image_ids = self.coco.getImgIds()
        self.logger.info(f"Found {len(self.image_ids)} images in {split} split")

    def __len__(self):
        return len(self.image_ids)

    def polygon_to_nodes_and_adjacency(self, segmentations):
        """Convert multiple COCO polygon segmentations to a unified node list and adjacency matrix"""
        all_nodes = []
        all_edges = []
        
        offset = 0  # Offset to index across multiple polygons
        for segmentation in segmentations:
            nodes = np.array(segmentation).reshape(-1, 2)  # Convert to (N,2) shape
            
            if len(nodes) == 0:
                continue  # Skip empty polygons
            
            all_nodes.append(nodes)

            # Create edges for the polygon
            num_nodes = len(nodes)
            for i in range(num_nodes):
                all_edges.append((offset + i, offset + (i + 1) % num_nodes))  # Connect consecutive nodes
            
            offset += num_nodes  # Update offset for next polygon
        
        if len(all_nodes) == 0:
            return np.zeros((self.max_nodes, 2)), np.zeros((self.max_nodes, self.max_nodes)), 0

        # Flatten node list
        nodes = np.vstack(all_nodes)  # (TotalNodes, 2)

        # Normalize coordinates to [0,1] range
        nodes = nodes / self.img_size

        # Create adjacency matrix
        num_nodes = len(nodes)
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        for i, j in all_edges:
            adjacency[i, j] = 1
            adjacency[j, i] = 1  # Make it symmetric
        
        return nodes, adjacency, num_nodes

    def pad_nodes(self, nodes, adjacency):
        num_nodes = len(nodes)

        if num_nodes > self.max_nodes:
            self.logger.warning(f"Truncating nodes from {num_nodes} to {self.max_nodes}")
            nodes = nodes[:self.max_nodes]
            adjacency = adjacency[:self.max_nodes, :self.max_nodes]
        elif num_nodes < self.max_nodes:
            nodes_pad = np.zeros((self.max_nodes - num_nodes, 2), dtype=np.float32)
            nodes = np.concatenate([nodes, nodes_pad], axis=0)

            adj_pad_rows = np.zeros((self.max_nodes - num_nodes, adjacency.shape[1]), dtype=np.float32)
            adjacency = np.concatenate([adjacency, adj_pad_rows], axis=0)
            adj_pad_cols = np.zeros((self.max_nodes, self.max_nodes - num_nodes), dtype=np.float32)
            adjacency = np.concatenate([adjacency, adj_pad_cols], axis=1)

        return nodes, adjacency, num_nodes

    def __getitem__(self, idx):
        try:
            # Get image info
            img_id = self.image_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]

            # Load image
            img_path = os.path.join(self.data_dir, self.split, 'images', img_info['file_name'])
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get annotations (all polygons for this image)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            if not anns:
                raise ValueError(f"No annotations found for image: {img_path}")

            # Extract all polygons from all annotations
            segmentations = [ann['segmentation'][0] for ann in anns if 'segmentation' in ann and ann['segmentation']]
            nodes, adjacency, num_nodes = self.polygon_to_nodes_and_adjacency(segmentations)

            # Pad or truncate nodes and adjacency
            nodes, adjacency, num_nodes = self.pad_nodes(nodes, adjacency)

            # Apply transforms
            if self.transform:
                image, nodes, adjacency = self.transform(image, nodes, adjacency)

            # Ensure image is in correct format (C, H, W)
            if len(image.shape) == 3 and image.shape[0] != 3:
                image = image.transpose(2, 0, 1)

            # Convert to tensor if needed
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float()
            if not isinstance(nodes, torch.Tensor):
                nodes = torch.from_numpy(nodes).float()
            if not isinstance(adjacency, torch.Tensor):
                adjacency = torch.from_numpy(adjacency).float()

            return image, nodes, adjacency, num_nodes

        except Exception as e:
            self.logger.error(f"Error loading sample {idx}: {str(e)}")
            raise

def collate_fn(batch):
    try:
        images, nodes, adjacencies, num_nodes = zip(*batch)
        images = torch.stack(images)
        nodes = torch.stack(nodes)
        adjacencies = torch.stack(adjacencies)
        num_nodes = torch.tensor(num_nodes)
        return images, nodes, adjacencies, num_nodes
    except Exception as e:
        logging.error(f"Error in collate_fn: {str(e)}")
        raise
