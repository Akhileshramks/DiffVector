import torch
from torch.utils.data import DataLoader
import argparse
import os
import logging

from models.diffvector import DiffVector
from dataset.remote_sensing import RemoteSensingDataset, collate_fn
from utils.transforms import RemoteSensingTransforms
from utils.visualization import VectorVisualizer
from training.trainer import DiffVectorTrainer

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DiffVector')

def parse_args():
    parser = argparse.ArgumentParser(description='DiffVector Training')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_timesteps', type=int, default=10)
    parser.add_argument('--max_nodes', type=int, default=32,
                      help='Maximum number of nodes per building')
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        # Verify data directory exists
        if not os.path.exists(args.data_dir):
            raise ValueError(f"Data directory {args.data_dir} does not exist")

        logger.info("Starting DiffVector training setup...")
        logger.info(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

        # Setup transforms and datasets
        transform = RemoteSensingTransforms(img_size=args.img_size)

        logger.info("Initializing datasets...")
        train_dataset = RemoteSensingDataset(
            args.data_dir,
            transform=transform,
            split='train',
            max_nodes=args.max_nodes
        )

        val_dataset = RemoteSensingDataset(
            args.data_dir,
            transform=transform,
            split='val',
            max_nodes=args.max_nodes
        )

        logger.info("Creating data loaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")

        # Initialize model
        logger.info("Initializing DiffVector model...")
        model = DiffVector(
            img_size=args.img_size,
            in_channels=3,
            embed_dim=96,
            num_heads=8,
            num_blocks=6,
            max_nodes=args.max_nodes
        )

        # Setup trainer
        logger.info("Setting up trainer...")
        trainer = DiffVectorTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=args.learning_rate,
            num_timesteps=args.num_timesteps
        )

        # Train model
        logger.info("Starting training...")
        trainer.train(args.num_epochs)

        # Visualization example
        logger.info("Generating visualization examples...")
        visualizer = VectorVisualizer()
        model.eval()

        with torch.no_grad():
            logger.info("Generating sample prediction visualization...")
            images, nodes, adjacency, num_nodes = next(iter(val_loader))
            outputs = model(images[:1])

            visualizer.draw_vectors(
                images[0],
                outputs['node_coords'][0][:num_nodes[0]],  # Only use valid nodes
                outputs['adjacency'][0][:num_nodes[0], :num_nodes[0]],  # Only use valid connections
                save_path='sample_prediction.png'
            )

            logger.info("Generating boundary maps visualization...")
            visualizer.plot_boundary_maps(
                outputs['boundary_maps'],
                save_path='boundary_maps.png'
            )

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()