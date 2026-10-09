"""Réseaux : générateur AnimeGAN (modèle avancé), générateur ResNet (baseline CycleGAN)
et discriminateur PatchGAN avec normalisation spectrale.

Tous les générateurs prennent et rendent un tenseur (N, 3, H, W) dans [-1, 1],
avec H et W multiples de 4 : c'est le contrat attendu par backend/app/inference.py.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class ConvBlock(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(
                c_in,
                c_out,
                kernel,
                stride,
                kernel // 2,
                groups=groups,
                bias=False,
                padding_mode="reflect",
            ),
            nn.GroupNorm(1, c_out),
            nn.LeakyReLU(0.2, inplace=True),
        )


class InvertedResBlock(nn.Module):
    """Bloc résiduel inversé (expansion, convolution en profondeur, projection)."""

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.body = nn.Sequential(
            ConvBlock(channels, hidden, kernel=1),
            ConvBlock(hidden, hidden, groups=hidden),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class Upsample(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = ConvBlock(c_in, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))


class AnimeGenerator(nn.Module):
    """Générateur léger de type AnimeGANv2 : rapide en inférence, adapté à l'export ONNX."""

    def __init__(self, base: int = 32, n_blocks: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, base, kernel=7),
            ConvBlock(base, base * 2, stride=2),
            ConvBlock(base * 2, base * 2),
            ConvBlock(base * 2, base * 4, stride=2),
            ConvBlock(base * 4, base * 4),
        )
        self.blocks = nn.Sequential(*[InvertedResBlock(base * 4) for _ in range(n_blocks)])
        self.decoder = nn.Sequential(
            Upsample(base * 4, base * 2),
            ConvBlock(base * 2, base * 2),
            Upsample(base * 2, base),
            ConvBlock(base, base),
            nn.Conv2d(base, 3, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.blocks(self.encoder(x)))


class ResnetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ResnetGenerator(nn.Module):
    """Générateur CycleGAN classique : sert de baseline (6 blocs pour du 128x128)."""

    def __init__(self, base: int = 64, n_blocks: int = 6):
        super().__init__()
        layers = [
            nn.Conv2d(3, base, 7, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),
        ]
        for mult in (1, 2):
            layers += [
                nn.Conv2d(base * mult, base * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(base * mult * 2),
                nn.ReLU(inplace=True),
            ]
        layers += [ResnetBlock(base * 4) for _ in range(n_blocks)]
        for mult in (4, 2):
            layers += [
                nn.ConvTranspose2d(
                    base * mult, base * mult // 2, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(base * mult // 2),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(base, 3, 7, padding=3, padding_mode="reflect"), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN : une note vrai/faux par région. La normalisation spectrale stabilise le jeu."""

    def __init__(self, base: int = 32, n_layers: int = 3):
        super().__init__()
        layers = [spectral_norm(nn.Conv2d(3, base, 3, padding=1)), nn.LeakyReLU(0.2, inplace=True)]
        channels = base
        for _ in range(n_layers - 1):
            layers += [
                spectral_norm(nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(channels * 2, channels * 4, 3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels *= 4
        layers += [
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(channels, 1, 3, padding=1)),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
