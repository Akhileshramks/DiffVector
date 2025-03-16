
import torch
import matplotlib.pyplot as plt
from models.diffvector import DiffVector
from utils.transforms import RemoteSensingTransforms
from utils.visualization import VectorVisualizer
import cv2
import numpy as np

def visualize_model_stages(image_path, model_path=None):
    # Initialize model
    model = DiffVector(img_size=224, in_channels=3, embed_dim=96, num_heads=8)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = RemoteSensingTransforms(img_size=224)
    image_tensor = transform(image).unsqueeze(0)
    
    # Visualizer
    visualizer = VectorVisualizer()
    
    with torch.no_grad():
        # 1. TCSwin Stage - Boundary Maps
        boundary_maps = model.tcswin(image_tensor)
        plt.figure(figsize=(15, 5))
        for i, bmap in enumerate(boundary_maps):
            plt.subplot(1, len(boundary_maps), i+1)
            plt.imshow(bmap[0].mean(dim=0).cpu().numpy())
            plt.title(f'Boundary Map {i}')
            plt.axis('off')
        plt.savefig('tcswin_output.png')
        plt.close()
        
        # 2. HiDiT Stage
        # Initialize node features and get last boundary map
        node_features = torch.zeros(1, model.max_nodes, model.embed_dim)
        last_boundary = boundary_maps[-1]
        last_boundary = last_boundary.flatten(2).transpose(1, 2)
        
        # Process through HiDiT
        node_features = model.hidit(node_features, torch.zeros(1), last_boundary)
        
        # Visualize node feature activations
        plt.figure(figsize=(10, 5))
        plt.imshow(node_features[0].cpu().numpy(), aspect='auto')
        plt.title('HiDiT Node Features')
        plt.colorbar()
        plt.savefig('hidit_output.png')
        plt.close()
        
        # 3. Node Coordinates and Edge Features
        # Get node coordinates
        node_coords = model.node_mlp(node_features)
        node_coords = torch.sigmoid(node_coords)
        
        # Get edge features
        edge_features = model.egdit.get_edge_embeddings(node_features)
        edge_features = edge_features.view(1, model.max_nodes, model.max_nodes, -1)
        
        # Visualize edge features
        plt.figure(figsize=(10, 10))
        plt.imshow(edge_features[0].mean(dim=-1).cpu().numpy())
        plt.title('Edge Features')
        plt.colorbar()
        plt.savefig('edge_features.png')
        plt.close()
        
        # 4. Final Output with Adjacency
        adjacency = model.egdit(node_features, edge_features.view(1, -1, edge_features.shape[-1]))
        
        # Visualize final output
        plt.figure(figsize=(10, 10))
        # Original image with predicted vectors
        plt.subplot(2, 2, 1)
        visualizer.draw_vectors(
            image_tensor[0],
            node_coords[0],
            torch.sigmoid(adjacency[0]),
            save_path=None
        )
        plt.title('Final Output')
        
        # Adjacency matrix
        plt.subplot(2, 2, 2)
        plt.imshow(torch.sigmoid(adjacency[0]).cpu().numpy())
        plt.title('Adjacency Matrix')
        
        plt.savefig('final_output.png')
        plt.close()

if __name__ == '__main__':
    # Test with a sample image
    visualize_model_stages('data/val/images/sample_000.jpg')
