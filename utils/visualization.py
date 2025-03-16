import matplotlib.pyplot as plt
import numpy as np
import torch

class VectorVisualizer:
    def __init__(self):
        self.colors = plt.cm.rainbow(np.linspace(0, 1, 20))

    def denormalize_image(self, image):
        """Convert normalized image back to [0, 255] range"""
        if torch.is_tensor(image):
            image = image.cpu().numpy()
        image = (image + 1.0) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def denormalize_coords(self, coords, img_size):
        """Convert normalized coordinates back to image space"""
        return coords * img_size

    def draw_vectors(self, image, nodes, adjacency, save_path=None):
        """Visualize building vectors on the image"""
        try:
            plt.figure(figsize=(10, 10))

            # Show image
            if torch.is_tensor(image) and image.shape[0] == 3:  # CHW to HWC                
                image = image.permute(1, 2, 0)
            image = self.denormalize_image(image)
            plt.imshow(image)

            # Draw nodes
            nodes = self.denormalize_coords(nodes, image.shape[0])
            plt.scatter(nodes[:, 0], nodes[:, 1], c='red', s=50)

            # Draw edges
            if torch.is_tensor(adjacency):
                adjacency = adjacency.cpu().numpy()

            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if adjacency[i, j] > 0.5:  # Threshold for edge presence
                        plt.plot([nodes[i, 0], nodes[j, 0]],
                               [nodes[i, 1], nodes[j, 1]],
                               c='blue', linewidth=2, alpha=0.6)

            plt.axis('off')
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"Error in draw_vectors: {str(e)}")
            print(f"Shapes: image={image.shape}, nodes={nodes.shape}, adjacency={adjacency.shape}")
            raise

    def plot_boundary_maps(self, boundary_maps, save_path=None):
        """Visualize boundary attention maps"""
        try:
            num_maps = len(boundary_maps)
            plt.figure(figsize=(4 * num_maps, 4))

            for i, bmap in enumerate(boundary_maps):
                if torch.is_tensor(bmap):
                    bmap = bmap.detach().cpu().numpy()
                plt.subplot(1, num_maps, i + 1)
                # Take mean across channels to get 2D heatmap
                bmap_2d = np.mean(bmap[0], axis=0)  # Average across channels
                plt.imshow(bmap_2d, cmap='viridis')
                plt.title(f'Boundary Map {i}')
                plt.axis('off')

            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"Error in plot_boundary_maps: {str(e)}")
            print(f"Number of boundary maps: {num_maps}")
            if boundary_maps:
                print(f"First boundary map shape: {boundary_maps[0].shape}")
            raise