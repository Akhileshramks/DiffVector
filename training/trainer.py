import torch
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import logging

class DiffVectorTrainer:
    def __init__(self, model, train_loader, val_loader=None,
                 lr=1e-4, num_timesteps=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_timesteps = num_timesteps

        self.optimizer = Adam(model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Setup logging
        self.logger = logging.getLogger('DiffVector')
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        try:
            with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
                for batch_idx, (images, nodes, adjacency, num_nodes) in enumerate(pbar):
                    try:
                        # Log batch information
                        self.logger.debug(f"Processing batch {batch_idx}:")
                        self.logger.debug(f"- images: {images.shape}, {images.dtype}")
                        self.logger.debug(f"- nodes: {nodes.shape}, {nodes.dtype}")
                        self.logger.debug(f"- adjacency: {adjacency.shape}, {adjacency.dtype}")
                        self.logger.debug(f"- num_nodes: {num_nodes}")

                        # Move batch to device
                        images = images.to(self.device)
                        nodes = nodes.to(self.device)
                        adjacency = adjacency.to(self.device)
                        num_nodes = num_nodes.to(self.device)

                        # Sample random timestep for each image
                        timesteps = torch.randint(
                            0, self.num_timesteps, (images.size(0),),
                            device=self.device
                        ).float()

                        # Forward pass with ITS
                        self.logger.debug("Starting forward pass")
                        loss, outputs = self.model.train_step(
                            images, timesteps, nodes, adjacency, num_nodes)

                        # Backward pass
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        # Log output information
                        self.logger.debug("Forward pass completed:")
                        self.logger.debug(f"- Loss: {loss.item():.4f}")
                        self.logger.debug(f"- Output shapes:")
                        for k, v in outputs.items():
                            if isinstance(v, torch.Tensor):
                                self.logger.debug(f"  {k}: {v.shape}")
                            elif isinstance(v, list):
                                self.logger.debug(f"  {k}: [List of {len(v)} tensors]")

                        # Update progress
                        total_loss += loss.item()
                        avg_loss = total_loss / (batch_idx + 1)
                        pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'avg_loss': f'{avg_loss:.4f}'
                        })

                    except RuntimeError as e:
                        self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                        self.logger.error("Batch shapes:")
                        self.logger.error(f"- images: {images.shape}")
                        self.logger.error(f"- nodes: {nodes.shape}")
                        self.logger.error(f"- adjacency: {adjacency.shape}")
                        self.logger.error(f"- num_nodes: {num_nodes.shape}")
                        raise

            avg_loss = total_loss / num_batches
            self.logger.info(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')
            return avg_loss

        except Exception as e:
            self.logger.error(f"Error during epoch {epoch}: {str(e)}")
            self.logger.error("Last successful batch shapes:")
            if 'images' in locals():
                self.logger.error(f"- images: {images.shape}")
            if 'nodes' in locals():
                self.logger.error(f"- nodes: {nodes.shape}")
            if 'adjacency' in locals():
                self.logger.error(f"- adjacency: {adjacency.shape}")
            raise

    @torch.no_grad()
    def validate(self):
        if not self.val_loader:
            return None

        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        try:
            for batch_idx, (images, nodes, adjacency, num_nodes) in enumerate(self.val_loader):
                try:
                    self.logger.debug(f"Validating batch {batch_idx}")

                    # Move batch to device
                    images = images.to(self.device)
                    nodes = nodes.to(self.device)
                    adjacency = adjacency.to(self.device)
                    num_nodes = num_nodes.to(self.device)

                    # Use zero timestep for validation (final prediction)
                    timesteps = torch.zeros(images.size(0), device=self.device)
                    loss, _ = self.model.train_step(images, timesteps, nodes, adjacency, num_nodes)
                    total_loss += loss.item()

                except RuntimeError as e:
                    self.logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    self.logger.error("Batch shapes:")
                    self.logger.error(f"- images: {images.shape}")
                    self.logger.error(f"- nodes: {nodes.shape}")
                    self.logger.error(f"- adjacency: {adjacency.shape}")
                    self.logger.error(f"- num_nodes: {num_nodes.shape}")
                    raise

            avg_loss = total_loss / num_batches
            self.logger.info(f'Validation Loss: {avg_loss:.4f}')
            return avg_loss

        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            raise

    def train(self, num_epochs):
        best_loss = float('inf')
        try:
            for epoch in range(num_epochs):
                self.logger.info(f"\nStarting epoch {epoch+1}/{num_epochs}")

                # Log memory usage before training
                if torch.cuda.is_available():
                    self.logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                    self.logger.info(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB")

                train_loss = self.train_epoch(epoch)
                val_loss = self.validate()

                # Save best model
                if val_loss and val_loss < best_loss:
                    best_loss = val_loss
                    self.logger.info(f"Saving best model with loss {best_loss:.4f}")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_loss,
                    }, 'best_model.pth')

        except Exception as e:
            self.logger.error(f"Training interrupted: {str(e)}")
            raise