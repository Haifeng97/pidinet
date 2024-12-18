"""
Author: Zhuo Su, Wenzhe Liu
Date: Feb 18, 2021
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import Conv2d
from .config import config_model, config_model_converted

class CSAM(nn.Module):
    """
    Compact Spatial Attention Module
    """
    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y

class CDCM(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)
        
    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4


class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """
    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock, self).__init__()
        self.stride=stride
            
        self.stride=stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class PDCBlock_converted(nn.Module):
    """
    CPDC, APDC can be converted to vanilla 3x3 convolution
    RPDC can be converted to vanilla 5x5 convolution
    """
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock_converted, self).__init__()
        self.stride=stride

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        if pdc == 'rd':
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=5, padding=2, groups=inplane, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class PiDiNet(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
        super(PiDiNet, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = []

        self.inplane = inplane
        if convert:
            if pdcs[0] == 'rd':
                init_kernel_size = 5
                init_padding = 2
            else:
                init_kernel_size = 3
                init_padding = 1
            self.init_block = nn.Conv2d(3, self.inplane, 
                    kernel_size=init_kernel_size, padding=init_padding, bias=False)
            block_class = PDCBlock_converted
        else:
            self.init_block = Conv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)
            block_class = PDCBlock

        self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # 2C
        
        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = block_class(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = block_class(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = block_class(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = block_class(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # 4C

        self.block4_1 = block_class(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = block_class(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = block_class(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = block_class(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # 4C

        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.attentions.append(CSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(4):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        self.classifier = nn.Conv2d(4, 1, kernel_size=1) # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

        print('initialization done')

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3, x4]

        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        e4 = self.conv_reduces[3](x_fuses[3])
        e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3, e4]

        output = self.classifier(torch.cat(outputs, dim=1))
        #if not self.training:
        #    return torch.sigmoid(output)

        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs


def pidinet_tiny(args):
    pdcs = config_model(args.config)
    dil = 8 if args.dil else None
    return PiDiNet(20, pdcs, dil=dil, sa=args.sa)

def pidinet_small(args):
    pdcs = config_model(args.config)
    dil = 12 if args.dil else None
    return PiDiNet(30, pdcs, dil=dil, sa=args.sa)

def pidinet(args):
    pdcs = config_model(args.config)
    dil = 24 if args.dil else None
    return PiDiNet(60, pdcs, dil=dil, sa=args.sa)



## convert pidinet to vanilla cnn

def pidinet_tiny_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 8 if args.dil else None
    return PiDiNet(20, pdcs, dil=dil, sa=args.sa, convert=True)

def pidinet_small_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 12 if args.dil else None
    return PiDiNet(30, pdcs, dil=dil, sa=args.sa, convert=True)

def pidinet_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 24 if args.dil else None
    return PiDiNet(60, pdcs, dil=dil, sa=args.sa, convert=True)


class PiDiNet_6x3(nn.Module):
    """
    6 stages, each stage 3 blocks, total 18 blocks + 1 init_block
    Stages: 0~5
    For example: inplane start from 30, every stride stage double channels
    """

    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
        super(PiDiNet_6x3, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        # total layers needed: 1 init + 18 blocks = 19 pdcs
        # pdcs[0] for init
        # pdcs[1..3] stage0
        # pdcs[4..6] stage1
        # pdcs[7..9] stage2
        # pdcs[10..12] stage3
        # pdcs[13..15] stage4
        # pdcs[16..18] stage5

        assert len(pdcs) >= 19, "Need at least 19 pdcs for 6x3 design"

        if convert:
            from .ops import PDCBlock_converted as PDCBlockClass
        else:
            PDCBlockClass = PDCBlock

        if convert:
            if pdcs[0] == 'rd':
                init_kernel_size = 5
                init_padding = 2
            else:
                init_kernel_size = 3
                init_padding = 1
            self.init_block = nn.Conv2d(3, inplane, kernel_size=init_kernel_size, padding=init_padding, bias=False)
        else:
            self.init_block = Conv2d(pdcs[0], 3, inplane, kernel_size=3, padding=1)

        # Stage definitions
        # Stage0 (3 blocks, no stride)
        self.stage0_1 = PDCBlockClass(pdcs[1], inplane, inplane, stride=1)
        self.stage0_2 = PDCBlockClass(pdcs[2], inplane, inplane, stride=1)
        self.stage0_3 = PDCBlockClass(pdcs[3], inplane, inplane, stride=1)
        fuseplanes = [inplane]

        # Stage1 (3 blocks, stride=2)
        inplane_stage1 = inplane
        inplane *= 2
        self.stage1_1 = PDCBlockClass(pdcs[4], inplane_stage1, inplane, stride=2)
        self.stage1_2 = PDCBlockClass(pdcs[5], inplane, inplane, stride=1)
        self.stage1_3 = PDCBlockClass(pdcs[6], inplane, inplane, stride=1)
        fuseplanes.append(inplane)

        # Stage2 (3 blocks, stride=2)
        inplane_stage2 = inplane
        inplane *= 2
        self.stage2_1 = PDCBlockClass(pdcs[7], inplane_stage2, inplane, stride=2)
        self.stage2_2 = PDCBlockClass(pdcs[8], inplane, inplane, stride=1)
        self.stage2_3 = PDCBlockClass(pdcs[9], inplane, inplane, stride=1)
        fuseplanes.append(inplane)

        # Stage3 (3 blocks, stride=2)
        inplane_stage3 = inplane
        inplane *= 2
        self.stage3_1 = PDCBlockClass(pdcs[10], inplane_stage3, inplane, stride=2)
        self.stage3_2 = PDCBlockClass(pdcs[11], inplane, inplane, stride=1)
        self.stage3_3 = PDCBlockClass(pdcs[12], inplane, inplane, stride=1)
        fuseplanes.append(inplane)

        # Stage4 (3 blocks, stride=2)
        inplane_stage4 = inplane
        inplane *= 2
        self.stage4_1 = PDCBlockClass(pdcs[13], inplane_stage4, inplane, stride=2)
        self.stage4_2 = PDCBlockClass(pdcs[14], inplane, inplane, stride=1)
        self.stage4_3 = PDCBlockClass(pdcs[15], inplane, inplane, stride=1)
        fuseplanes.append(inplane)

        # Stage5 (3 blocks, stride=2)
        inplane_stage5 = inplane
        inplane *= 2
        self.stage5_1 = PDCBlockClass(pdcs[16], inplane_stage5, inplane, stride=2)
        self.stage5_2 = PDCBlockClass(pdcs[17], inplane, inplane, stride=1)
        self.stage5_3 = PDCBlockClass(pdcs[18], inplane, inplane, stride=1)
        fuseplanes.append(inplane)

        self.fuseplanes = fuseplanes

        # Attention / Dilation / Reduce
        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for fp in fuseplanes:
                self.dilations.append(CDCM(fp, self.dil))
                self.attentions.append(CSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for fp in fuseplanes:
                self.attentions.append(CSAM(fp))
                self.conv_reduces.append(MapReduce(fp))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for fp in fuseplanes:
                self.dilations.append(CDCM(fp, self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for fp in fuseplanes:
                self.conv_reduces.append(MapReduce(fp))

        self.classifier = nn.Conv2d(len(fuseplanes), 1, kernel_size=1)  # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

        print('6x3 initialization done')

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)
        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        H, W = x.size()[2:]
        x = self.init_block(x)

        # stage0
        x0 = self.stage0_1(x)
        x0 = self.stage0_2(x0)
        x0 = self.stage0_3(x0)
        # stage1
        x1 = self.stage1_1(x0)
        x1 = self.stage1_2(x1)
        x1 = self.stage1_3(x1)
        # stage2
        x2 = self.stage2_1(x1)
        x2 = self.stage2_2(x2)
        x2 = self.stage2_3(x2)
        # stage3
        x3 = self.stage3_1(x2)
        x3 = self.stage3_2(x3)
        x3 = self.stage3_3(x3)
        # stage4
        x4 = self.stage4_1(x3)
        x4 = self.stage4_2(x4)
        x4 = self.stage4_3(x4)
        # stage5
        x5 = self.stage5_1(x4)
        x5 = self.stage5_2(x5)
        x5 = self.stage5_3(x5)

        features = [x0, x1, x2, x3, x4, x5]

        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate(features):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate(features):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate(features):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = features

        outs = []
        for i, xf in enumerate(x_fuses):
            ei = self.conv_reduces[i](xf)
            ei = F.interpolate(ei, (H, W), mode="bilinear", align_corners=False)
            outs.append(ei)

        output = self.classifier(torch.cat(outs, dim=1))
        outs.append(output)
        outs = [torch.sigmoid(r) for r in outs]
        return outs

def pidinet_6x3(args):
    pdcs = config_model(args.config)  # 请确保config_model返回至少19个pdcs
    dil = 24 if args.dil else None
    # 假设初始通道数为30（可根据需求修改）
    return PiDiNet_6x3(30, pdcs, dil=dil, sa=args.sa)