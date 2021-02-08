import torch
import torch.nn as nn
import math

class ConvLayer2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=padding, dilation=dilation, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, kernel_size, stride, dilation_list, in_size):
        super().__init__()
        if len(dilation_list) < n_layers:
            dilation_list = dilation_list + [dilation_list[-1]] * (n_layers - len(dilation_list))

        padding = []
        # Compute padding for each temporal layer to have a fixed size output
        # Output size is controlled by striding to be 1 / 'striding' of the original size
        for dilation in dilation_list:
            filter_size = kernel_size[1] * dilation[1] - 1
            temp_pad = math.floor((filter_size - 1) / 2) - 1 * (dilation[1] // 2 - 1)
            padding.append((0, temp_pad))

        self.layers = nn.ModuleList([
            ConvLayer2D(
                in_channels, out_channels, kernel_size, stride, padding[i], dilation_list[i]
            ) for i in range(n_layers)
        ])

    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)
        return out

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_spatial_layers, stride, input_height):
        super().__init__()
       
        kernel_list = []
        for i in range(num_spatial_layers):
            kernel_list.append(((input_height // (i + 1)), 1))

        padding = []
        for kernel in kernel_list:
            temp_pad = math.floor((kernel[0] - 1) / 2)# - 1 * (kernel[1] // 2 - 1)
            padding.append((temp_pad, 0))

        feature_height = input_height // stride[0]

        self.layers = nn.ModuleList([
            ConvLayer2D(
                in_channels, out_channels, kernel_list[i], stride, padding[i], 1
            ) for i in range(num_spatial_layers)
        ])
    
    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)

        return out

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out