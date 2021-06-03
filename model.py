import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self.block(in_channels=features, out_channels=features*2, kernel_size=4, stride=2, padding=1),
            self.block(in_channels=features*2, out_channels=features*4, kernel_size=4, stride=2, padding=1),
            self.block(in_channels=features*4, out_channels=features*8, kernel_size=4, stride=2, padding=1),
            self.block(in_channels=features*8, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, noise_dim, channels, features):
        super().__init__()
        self.gen = nn.Sequential(
            self.block(in_channels=noise_dim, out_channels=features*16, kernel_size=4, stride=2, padding=0),
            self.block(features*16, features*8, kernel_size=4, stride=2, padding=1),
            self.block(features*8, features*4, kernel_size=4, stride=2, padding=1),
            self.block(features*4, features*2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(features*2, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(layer.weight.data, 0, 0.002)
