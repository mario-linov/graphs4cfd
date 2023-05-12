import torch.nn as nn
import torch.nn.functional as F


class GraphLoss(nn.Module):
    def __init__(self, lambda_d=0):
        super().__init__()
        self.lambda_d  = lambda_d

    def forward(self, graph, pred, target):
        loss = F.mse_loss(pred, target)
        if self.lambda_d > 0:
            dirichlet_boundary = (graph.omega[:,0] == 1)
            if dirichlet_boundary.any():
                loss += self.lambda_d*F.l1_loss(pred[dirichlet_boundary], target[dirichlet_boundary])
        return loss
