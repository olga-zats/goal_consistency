import torch
import torch.nn.functional as F

class LatentConsistencyLoss(torch.nn.Module):
    def __init__(self, cond_prob):
        super().__init__()
        self.cond_prob = cond_prob

    def forward(self, x_p, z_p, weight=None, val=False):
        hat_z_p = torch.sum(self.cond_prob[None, :] * x_p[..., None], dim=1)  # B x C_z
        hat_z_p = torch.clamp(hat_z_p, min=1e-5)  # B x C_z
        loss = -torch.sum(z_p * torch.log(hat_z_p), dim=-1)  # B

        if weight is not None:
            loss = loss * weight

        if val:
            loss = loss[~loss.isnan()]
            if loss.mean().isnan():
                loss = torch.as_tensor([0.0])
        return loss.mean()

