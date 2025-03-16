import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss:
    def __init__(self, node_weight=1.0, adjacency_weight=1.0):
        self.node_weight = node_weight
        self.adjacency_weight = adjacency_weight
        
    def node_loss(self, pred_nodes, gt_nodes):
        """MSE loss for node positions"""
        return F.mse_loss(pred_nodes, gt_nodes)
    
    def adjacency_loss(self, pred_adj, gt_adj):
        """Binary cross-entropy loss for adjacency matrix"""
        return F.binary_cross_entropy_with_logits(pred_adj, gt_adj)
    
    def boundary_loss(self, pred_maps, gt_maps):
        """L1 loss for boundary attention maps"""
        return sum(F.l1_loss(p, g) for p, g in zip(pred_maps, gt_maps))
    
    def __call__(self, predictions, targets):
        node_loss = self.node_loss(
            predictions['node_coords'],
            targets['nodes']
        ) * self.node_weight
        
        adj_loss = self.adjacency_loss(
            predictions['adjacency'],
            targets['adjacency']
        ) * self.adjacency_weight
        
        total_loss = node_loss + adj_loss
        
        return {
            'total': total_loss,
            'node': node_loss.item(),
            'adjacency': adj_loss.item()
        }
