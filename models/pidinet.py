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


class PiDiNet_6x4(nn.Module):
    def __init__(self, pdcs, dil, sa, convert=False):
        super(PiDiNet_6x4, self).__init__()
        # 前3个stage 60通道，第4个120通道，第5个240，第6个480
        # 共24个block + 1 init block = 25 ops
        assert len(pdcs) == 25, "Need 25 pdcs for 6x4 design"
        self.sa = sa
        self.dil = dil
        assert self.sa and self.dil is not None, "Assume sa and dil both True"
        if convert:
            block_class = PDCBlock_converted
        else:
            block_class = PDCBlock

        # 通道设置
        c1 = 100
        c2 = 100
        c3 = 100
        c4 = 200
        c5 = 400
        c6 = 800

        pdc_init = pdcs[0]
        self.init_block = Conv2d(pdc_init, 3, c1, kernel_size=3, padding=1)

        # 定义stage构建函数
        # stage输入参数：start_idx对应pdcs起始index（从1开始，因为0是init）
        # stage包含4个block
        def make_stage(start_idx, inplane, outplane, downsample=False):
            # 如果downsample=True，则第一个block stride=2，否则stride=1
            stride = 2 if downsample else 1
            blocks = []
            blocks.append(block_class(pdcs[start_idx],   inplane, outplane, stride=stride))
            blocks.append(block_class(pdcs[start_idx+1], outplane, outplane))
            blocks.append(block_class(pdcs[start_idx+2], outplane, outplane))
            blocks.append(block_class(pdcs[start_idx+3], outplane, outplane))
            return nn.Sequential(*blocks)

        # 注意pdcs[1..24]
        # stage1 (4个block): 通道不变c1=60，不下采样
        self.stage1 = make_stage(1, c1, c1, downsample=False)
        # stage2 (4个block): 仍然是60通道，但这里可以不下采样
        # 但题目要求前三个stage都是60通道，并无下采样，只有到第四个stage才加通道吗？
        # 要求是前三个stage都是60通道。那意味着stage2, stage3也不下采样且通道不变
        self.stage2 = make_stage(5, c1, c1, downsample=False)
        self.stage3 = make_stage(9, c1, c1, downsample=False)

        # 第四个stage c1->c4=120 通道增加并下采样
        self.stage4 = make_stage(13, c1, c4, downsample=True)
        # 第五个stage c4->c5=240 通道增加并下采样
        self.stage5 = make_stage(17, c4, c5, downsample=True)
        # 第六个stage c5->c6=480 通道增加并下采样
        self.stage6 = make_stage(21, c5, c6, downsample=True)

        # fuseplanes记录每个stage输出通道数
        self.fuseplanes = [c1, c1, c1, c4, c5, c6]

        self.attentions = nn.ModuleList()
        self.dilations = nn.ModuleList()
        self.conv_reduces = nn.ModuleList()

        for p in self.fuseplanes:
            self.dilations.append(CDCM(p, self.dil))
            self.attentions.append(CSAM(self.dil))
            self.conv_reduces.append(MapReduce(self.dil))

        self.classifier = nn.Conv2d(6, 1, kernel_size=1)
        nn.init.constant_(self.classifier.weight, 1.0/6.0)
        nn.init.constant_(self.classifier.bias, 0)
        print('6x4 initialization done')

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

        x1 = self.stage1(x)  # 60通道
        x2 = self.stage2(x1) # 60通道
        x3 = self.stage3(x2) # 60通道
        x4 = self.stage4(x3) # 120通道
        x5 = self.stage5(x4) # 240通道
        x6 = self.stage6(x5) # 480通道

        xs = [x1, x2, x3, x4, x5, x6]
        e_list = []
        for i, xi in enumerate(xs):
            ei = self.attentions[i](self.dilations[i](xi))
            ei = self.conv_reduces[i](ei)
            ei = F.interpolate(ei, (H, W), mode="bilinear", align_corners=False)
            e_list.append(ei)

        out_cat = torch.cat(e_list, dim=1)
        output = self.classifier(out_cat)
        outputs = e_list + [output]
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs

def pidinet6x4(args):
    pdcs = config_model(args.config) # 确保config中定义carv_6x4共25个pdc
    dil = 24 if args.dil else None
    return PiDiNet_6x4(pdcs, dil=dil, sa=args.sa)