import os
import numpy as np
import cv2
from zipfile import ZipFile
from io import BytesIO
import shutil

def download_and_prepare_data():
    """Download and prepare sample satellite imagery dataset"""
    # Create data directories if they don't exist
    os.makedirs('data/train/images', exist_ok=True)
    os.makedirs('data/train/annotations', exist_ok=True)
    os.makedirs('data/val/images', exist_ok=True)
    os.makedirs('data/val/annotations', exist_ok=True)

    # Generate synthetic data
    num_train = 10
    num_val = 2

    # Generate training data
    for i in range(num_train):
        # Create synthetic satellite image
        image = np.ones((224, 224, 3), dtype=np.uint8) * 200  # Light gray background

        # Add random building shapes
        num_buildings = np.random.randint(1, 4)
        buildings = []

        for _ in range(num_buildings):
            # Random building size and position
            width = np.random.randint(30, 80)
            height = np.random.randint(30, 80)
            x = np.random.randint(0, 224 - width)
            y = np.random.randint(0, 224 - height)

            # Create building polygon
            building = np.array([
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height]
            ], dtype=np.int32)
            buildings.append(building)

            # Draw building
            cv2.fillPoly(image, [building], (100, 100, 100))  # Dark gray building
            cv2.polylines(image, [building], True, (80, 80, 80), 2)  # Building outline

        # Add texture and noise
        noise = np.random.normal(0, 10, image.shape).astype(np.int32)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

        # Add some random vegetation-like features
        for _ in range(20):
            x = np.random.randint(0, 224)
            y = np.random.randint(0, 224)
            cv2.circle(image, (x, y), np.random.randint(2, 5), 
                      (70, 120, 70), -1)  # Green dots for vegetation

        # Save image
        cv2.imwrite(f'data/train/images/sample_{i:03d}.jpg', image)

        # Save annotations
        nodes = np.concatenate([building.astype(np.float32) for building in buildings])
        adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)

        # Create adjacency matrix for each building
        start_idx = 0
        for building in buildings:
            size = len(building)
            for j in range(size):
                adjacency[start_idx + j, start_idx + (j + 1) % size] = 1
                adjacency[start_idx + (j + 1) % size, start_idx + j] = 1
            start_idx += size

        np.savez(f'data/train/annotations/sample_{i:03d}.npz',
                 nodes=nodes, adjacency=adjacency)

    # Generate validation data
    for i in range(num_val):
        # Create synthetic satellite image
        image = np.ones((224, 224, 3), dtype=np.uint8) * 200

        # Single building for validation
        width = np.random.randint(40, 60)
        height = np.random.randint(40, 60)
        x = np.random.randint(0, 224 - width)
        y = np.random.randint(0, 224 - height)

        building = np.array([
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height]
        ], dtype=np.int32)

        cv2.fillPoly(image, [building], (100, 100, 100))
        cv2.polylines(image, [building], True, (80, 80, 80), 2)

        # Add texture and noise
        noise = np.random.normal(0, 10, image.shape).astype(np.int32)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

        # Save image
        cv2.imwrite(f'data/val/images/sample_{i:03d}.jpg', image)

        # Save annotation
        nodes = building.astype(np.float32)
        adjacency = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=np.float32)

        np.savez(f'data/val/annotations/sample_{i:03d}.npz',
                 nodes=nodes, adjacency=adjacency)

    print("Generated synthetic dataset successfully!")
    print(f"Created {num_train} training samples and {num_val} validation samples")

if __name__ == '__main__':
    download_and_prepare_data()