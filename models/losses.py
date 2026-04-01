import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x = sobel_x.repeat(3, 1, 1, 1).to(device)
        self.sobel_y = sobel_y.repeat(3, 1, 1, 1).to(device)
        self.loss = nn.L1Loss()
    def forward(self, pred, target):
        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        grad_target_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        loss_x = self.loss(grad_pred_x, grad_target_x)
        loss_y = self.loss(grad_pred_y, grad_target_y)
        return loss_x + loss_y

class ResidualGradientLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x = sobel_x.repeat(4, 1, 1, 1).to(device)
        self.sobel_y = sobel_y.repeat(4, 1, 1, 1).to(device)
        self.loss = nn.L1Loss()
    def forward(self, residual_map):
        grad_res_x = F.conv2d(residual_map, self.sobel_x, padding=1, groups=4)
        grad_res_y = F.conv2d(residual_map, self.sobel_y, padding=1, groups=4)
        return self.loss(grad_res_x, torch.zeros_like(grad_res_x)) + self.loss(grad_res_y, torch.zeros_like(grad_res_y))


def flow_smoothness_loss(flow):
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    return torch.mean(dx) + torch.mean(dy)
