import numpy as np
import cv2
import os
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DataGenerator')

def generate_building(img_size=224, min_size=30, max_size=80):
    """Generate a single building with random dimensions"""
    try:
        # Generate random building parameters
        width = np.random.randint(min_size, max_size)
        height = np.random.randint(min_size, max_size)
        x = np.random.randint(0, img_size - width)
        y = np.random.randint(0, img_size - height)

        # Create building corners (nodes)
        nodes = np.array([
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height]
        ], dtype=np.float32)

        # Create adjacency matrix (connecting neighboring nodes)
        adjacency = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=np.float32)

        return nodes, adjacency

    except Exception as e:
        logger.error(f"Error generating building: {str(e)}")
        raise

def main():
    """Generate synthetic dataset"""
    try:
        # Create required directories
        os.makedirs('data/train/images', exist_ok=True)
        os.makedirs('data/train/annotations', exist_ok=True)
        os.makedirs('data/val/images', exist_ok=True)
        os.makedirs('data/val/annotations', exist_ok=True)

        # Dataset parameters
        num_train = 10
        num_val = 2
        img_size = 224

        logger.info("Generating training data...")
        for i in tqdm(range(num_train), desc="Training samples"):
            # Generate synthetic sample
            image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200  # Light gray background
            nodes, adjacency = generate_building(img_size)

            # Draw building
            pts = nodes.astype(np.int32)
            cv2.fillPoly(image, [pts], (100, 100, 100))  # Fill with dark gray
            cv2.polylines(image, [pts], True, (80, 80, 80), 2)  # Add border

            # Add noise and texture
            noise = np.random.normal(0, 10, image.shape).astype(np.int32)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

            # Add random vegetation-like features
            for _ in range(20):
                x = np.random.randint(0, img_size)
                y = np.random.randint(0, img_size)
                radius = np.random.randint(2, 5)
                cv2.circle(image, (x, y), radius, (70, 120, 70), -1)

            # Save image and annotation
            cv2.imwrite(f'data/train/images/sample_{i:03d}.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            np.savez(f'data/train/annotations/sample_{i:03d}.npz', nodes=nodes, adjacency=adjacency)

        logger.info("\nGenerating validation data...")
        for i in tqdm(range(num_val), desc="Validation samples"):
            # Generate synthetic sample
            image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200
            nodes, adjacency = generate_building(img_size)

            # Draw building
            pts = nodes.astype(np.int32)
            cv2.fillPoly(image, [pts], (100, 100, 100))
            cv2.polylines(image, [pts], True, (80, 80, 80), 2)

            # Add noise and texture
            noise = np.random.normal(0, 10, image.shape).astype(np.int32)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

            # Save image and annotation
            cv2.imwrite(f'data/val/images/sample_{i:03d}.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            np.savez(f'data/val/annotations/sample_{i:03d}.npz', nodes=nodes, adjacency=adjacency)

        logger.info("\nDataset generation completed successfully!")
        logger.info(f"Created {num_train} training samples and {num_val} validation samples")

    except Exception as e:
        logger.error(f"Error generating dataset: {str(e)}")
        raise

if __name__ == '__main__':
    main()