# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class NONLocalBlock1D(nn.Module):
    def __init__(self, args, recent_dim, spanning_dim, latent_dim):
        super(NONLocalBlock1D, self).__init__()

        self.in_dim1 = recent_dim
        self.in_dim2 = spanning_dim

        self.scale = args.scale
        self.scale_factor = args.scale_factor

        self.dropout_rate = args.dropout_rate

        self.latent_dim = latent_dim
        self.video_feat_dim = args.video_feat_dim

        self.theta = nn.Conv1d(in_channels=self.in_dim1, out_channels=self.latent_dim,
                               kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.theta.weight, mean=0, std=0.01)
        nn.init.constant_(self.theta.bias, 0)

        self.phi = nn.Conv1d(in_channels=self.in_dim2, out_channels=self.latent_dim,
                             kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.phi.weight, mean=0, std=0.01)
        nn.init.constant_(self.phi.bias, 0)

        self.g = nn.Conv1d(in_channels=self.in_dim2, out_channels=self.latent_dim,
                           kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.g.weight, mean=0, std=0.01)
        nn.init.constant_(self.g.bias, 0)

        if self.scale:
            self.scale_factor = torch.tensor([self.latent_dim ** self.scale_factor], requires_grad=True).to('cuda') # to get GFLOPs set to requires_grad=False

        # """Pre-activation style non-linearity."""
        self.final_layers = nn.Sequential(
            nn.LayerNorm(torch.Size([self.latent_dim, self.video_feat_dim])),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.latent_dim, out_channels=self.in_dim1, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=self.dropout_rate),
        )

    def forward(self, x_past, x_curr):
        theta_x = self.theta(x_curr)
        
        phi_x = self.phi(x_past)
        phi_x = phi_x.permute(0, 2, 1)

        g_x = self.g(x_past)
        
        theta_phi = torch.matmul(theta_x, phi_x)    # (NxCxnum_feat1) X (NxCxnum_feat2) = (NxCxC)

        if self.scale:
            theta_phi = theta_phi * self.scale_factor

        p_x = F.softmax(theta_phi, dim=-1)
        
        t_x = torch.matmul(p_x, g_x)                # (NxCxC) X (NxCxnum_feat2) = (BxCxnum_feat2)
                
        W_t = self.final_layers(t_x)

        z_x = W_t + x_curr
        return z_x
