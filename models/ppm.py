"""
Author: Haifeng Jia
Date: Nov 28, 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.stages = []
        for pool_size in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1),
                nn.ReLU(inplace=True)
            ))
        self.stages = nn.ModuleList(self.stages)
        self.bottleneck = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.size()[2:]
        pyramids = [x]
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            pyramids.append(out)
        output = torch.cat(pyramids, dim=1)
        output = self.relu(self.bottleneck(output))
        return output
