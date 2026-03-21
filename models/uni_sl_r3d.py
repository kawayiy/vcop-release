import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class SpatioTemporalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super().__init__()
        self.downsample = downsample
        padding = kernel_size // 2

        if self.downsample:
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
            self.conv1 = SpatioTemporalConv(
                in_channels, out_channels, kernel_size, padding=padding, stride=2
            )
        else:
            self.conv1 = SpatioTemporalConv(
                in_channels, out_channels, kernel_size, padding=padding
            )

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        layer_size,
        block_type=SpatioTemporalResBlock,
        downsample=False,
    ):
        super().__init__()

        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        self.blocks = nn.ModuleList()
        for _ in range(layer_size - 1):
            self.blocks.append(block_type(out_channels, out_channels, kernel_size))

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x


class UniSLR3D(nn.Module):
    def __init__(
        self,
        layer_sizes=(1, 1, 1, 1),
        block_type=SpatioTemporalResBlock,
        with_classifier=False,
        return_conv=False,
        num_classes=101,
    ):
        super().__init__()
        self.with_classifier = with_classifier
        self.return_conv = return_conv
        self.num_classes = num_classes
        self.output_dim = 512

        self.conv1 = SpatioTemporalConv(
            3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3]
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = SpatioTemporalResLayer(
            64, 64, 3, layer_sizes[0], block_type=block_type
        )
        self.conv3 = SpatioTemporalResLayer(
            64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True
        )
        self.conv4 = SpatioTemporalResLayer(
            128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True
        )
        self.conv5 = SpatioTemporalResLayer(
            256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True
        )

        if self.return_conv:
            self.feature_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.pool = nn.AdaptiveAvgPool3d(1)

        if self.with_classifier:
            self.linear = nn.Linear(self.output_dim, self.num_classes)

    def forward_features(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if self.return_conv:
            x = self.feature_pool(x)
            return x.view(x.shape[0], -1)

        x = self.pool(x)
        x = x.view(x.shape[0], self.output_dim)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.with_classifier:
            x = self.linear(x)

        return x


if __name__ == "__main__":
    model = UniSLR3D(layer_sizes=(1, 1, 1, 1), with_classifier=False)
    x = torch.randn(2, 3, 16, 112, 112)
    y = model(x)
    print("output shape:", y.shape)