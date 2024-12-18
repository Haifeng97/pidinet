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


class PiDiNetDeeperNarrow(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False):
        super(PiDiNetDeeperNarrow, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be int'
        self.dil = dil

        self.fuseplanes = []
        # 假设我们用6个stage，每个stage 2个block，共12个block
        # 通道数保持不变，如全部为inplane=30
        # 每隔2个stage下采样一次（类似原piDiNet的逻辑）

        self.inplane = inplane
        # 初始化
        # 假设第一个pdc是pdcs[0]，kernel仍然用同样的逻辑
        self.init_block = nn.Conv2d(3, self.inplane, kernel_size=3, padding=1, bias=False)

        block_class = PDCBlock
        # block布局：
        # stage1: 2个block，不下采样
        # stage2: 2个block，下采样
        # stage3: 2个block，不下采样
        # stage4: 2个block，下采样
        # stage5: 2个block，不下采样
        # stage6: 2个block，下采样（可选择）

        # 为简单起见，我们使用pdcs依次取，这里有12个block需要12个pdc：
        # 如果pdcs不够，可自行扩展config文件
        idx = 0

        def make_stage(pdc_list, in_c, out_c, stride):
            layers = []
            for i, pdc in enumerate(pdc_list):
                s = stride if i == 0 else 1
                layers.append(block_class(pdc, in_c, out_c, stride=s))
            return nn.Sequential(*layers)

        # 构建6个stage，每个stage 2个block
        # 通道数固定为inplane，不扩大
        # 每隔一个stage下采样一次，这里假设stage2、stage4、stage6下采样
        self.stage1 = make_stage(pdcs[idx:idx + 2], self.inplane, self.inplane, stride=1)
        idx += 2
        self.fuseplanes.append(self.inplane)

        self.stage2 = make_stage(pdcs[idx:idx + 2], self.inplane, self.inplane, stride=2)
        idx += 2
        self.fuseplanes.append(self.inplane)

        self.stage3 = make_stage(pdcs[idx:idx + 2], self.inplane, self.inplane, stride=1)
        idx += 2
        self.fuseplanes.append(self.inplane)

        self.stage4 = make_stage(pdcs[idx:idx + 2], self.inplane, self.inplane, stride=2)
        idx += 2
        self.fuseplanes.append(self.inplane)

        self.stage5 = make_stage(pdcs[idx:idx + 2], self.inplane, self.inplane, stride=1)
        idx += 2
        self.fuseplanes.append(self.inplane)

        self.stage6 = make_stage(pdcs[idx:idx + 2], self.inplane, self.inplane, stride=2)
        idx += 2
        self.fuseplanes.append(self.inplane)

        # 构建融合层与classifier
        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(6):  # 6个stage输出
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.attentions.append(CSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(6):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(6):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(6):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        # 最终分类器，需要聚合6个map
        self.classifier = nn.Conv2d(6, 1, kernel_size=1)
        nn.init.constant_(self.classifier.weight, 1.0 / 6)
        nn.init.constant_(self.classifier.bias, 0)

        print('DeeperNarrow initialization done')

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
        # 前向传播6个stage
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        x6 = self.stage6(x5)

        # attention与dilation
        features = [x1, x2, x3, x4, x5, x6]
        x_fuses = []
        for i, xi in enumerate(features):
            if self.sa and self.dil is not None:
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
            elif self.sa:
                x_fuses.append(self.attentions[i](xi))
            elif self.dil is not None:
                x_fuses.append(self.dilations[i](xi))
            else:
                x_fuses.append(xi)

        # reduce到1通道并上采样
        outputs = []
        for i, xf in enumerate(x_fuses):
            e = self.conv_reduces[i](xf)
            e = F.interpolate(e, (H, W), mode="bilinear", align_corners=False)
            outputs.append(e)

        # 聚合
        output = self.classifier(torch.cat(outputs, dim=1))
        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs

def pidinet_deeper_narrow(args):
    # 假设使用carv4这个配置，可自行更换
    pdcs = config_model(args.config)
    dil = 8 if args.dil else None
    # 减少初始通道数，如30
    return PiDiNetDeeperNarrow(inplane=30, pdcs=pdcs, dil=dil, sa=args.sa)