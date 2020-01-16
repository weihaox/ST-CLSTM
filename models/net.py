import torch
import torch.nn as nn
from models import modules
from models.refinenet_dict import refinenet_dict


def cubes_2_maps(cubes):
    b, c, d, h, w = cubes.shape
    cubes = cubes.permute(0, 2, 1, 3, 4)

    return cubes.contiguous().view(b*d, c, h, w), b, d


class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel, refinenet):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = refinenet_dict[refinenet](block_channel)

    def forward(self, x):
        x, b, d = cubes_2_maps(x)
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1), b, d)

        return out

class C_C3D_1(nn.Module):

    def __init__(self, num_classes=2, init_width=32, input_channels=1):
        self.inplanes = init_width
        super(C_C3D_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, init_width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(init_width),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(init_width, init_width*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(init_width*2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        init_width = init_width * 2
        self.conv3 = nn.Sequential(
            nn.Conv3d(init_width, init_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(init_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        init_width = init_width * 2
        self.conv4 = nn.Sequential(
            nn.Conv3d(init_width, init_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(init_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(init_width * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class C_C3D_2(nn.Module):

    def __init__(self, num_classes=2, init_width=32, input_channels=1):
        self.inplanes = init_width
        super(C_C3D_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, init_width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(init_width),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(init_width, init_width*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(init_width*2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        init_width = init_width * 2
        self.conv3 = nn.Sequential(
            nn.Conv3d(init_width, init_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(init_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(init_width * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x