"""
Author: Haifeng Jia
Date: Nov 27, 2024
"""

import torch
import torch.nn as nn


# 在 pidinet.py 文件中添加

class SelfAttention(nn.Module):
    """
    Self-Attention Module
    """

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, H * W)  # B x C' x N
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)  # B x C' x N
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # B x N x N
        attention = self.softmax(energy)  # B x N x N
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)  # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, H, W)
        out = self.gamma * out + x
        return out
