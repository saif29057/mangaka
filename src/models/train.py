"""Entraînement AnimeGAN (modèle avancé) avec suivi MLflow.

Exemple :
    python -m src.data.dataset --style Hayao
    python -m src.models.train --style Hayao --epochs 30 --init-epochs 2

Déroulement :
  1. Phase d'initialisation : le générateur apprend seulement à reconstruire la photo
     (perte de contenu), ce qui évite un départ instable du jeu adverse.
  2. Phase adverse : pertes adverse (LSGAN), contenu, style (matrices de Gram), couleur.

TODO Membre A :
  - baseline CycleGAN (ResnetGenerator x2 + PatchDiscriminator x2, perte de cycle) ;
  - reprise depuis des poids pré-entraînés (--resume) pour le fine-tuning ;
  - ajuster les poids des pertes par style (les valeurs par défaut sont un point de départ).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import VGG19_Weights, vgg19

from src.data.dataset import AnimeDataset
from src.models.architecture import AnimeGenerator, PatchDiscriminator
from src.utils.visualization import comparison_grid


class VGGFeatures(nn.Module):
    """Caractéristiques VGG19 (jusqu'à conv4_4) pour les pertes perceptuelles."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = vgg19(weights=weights).features[:26].eval()
        for param in self.parameters():
            param.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(((x + 1.0) / 2.0 - self.mean) / self.std)


def gram(features: torch.Tensor) -> torch.Tensor:
    n, c, h, w = features.shape
    flat = features.view(n, c, h * w)
    return flat @ flat.transpose(1, 2) / (c * h * w)


def to_gray3(x: torch.Tensor) -> torch.Tensor:
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return gray.repeat(1, 3, 1, 1)


def rgb_to_yuv(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return torch.stack([y, 0.492 * (b - y), 0.877 * (r - y)], dim=1)


def color_loss(fake: torch.Tensor, photo: torch.Tensor) -> torch.Tensor:
    """Garde les couleurs de la photo : L1 sur la luminance, Huber sur la chrominance."""
    fake_yuv, photo_yuv = rgb_to_yuv(fake), rgb_to_yuv(photo)
    return F.l1_loss(fake_yuv[:, 0], photo_yuv[:, 0]) + F.smooth_l1_loss(
        fake_yuv[:, 1:], photo_yuv[:, 1:]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement AnimeGAN")
    parser.add_argument("--style", required=True, help="Hayao ou Shinkai")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--init-epochs", type=int, default=2)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=4e-4)
    parser.add_argument("--w-adv", type=float, default=1.0)
    parser.add_argument("--w-con", type=float, default=1.5)
    parser.add_argument("--w-gram", type=float, default=3.0)
    parser.add_argument("--w-col", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0, help="Limite par époque (debug)")
    parser.add_argument("--no-pretrained-vgg", action="store_true", help="Tests hors ligne")
    parser.add_argument("--experiment", default="anime-style-transfer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AnimeDataset(args.raw_dir, args.processed_dir, args.style, args.size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )

    generator = AnimeGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    vgg = VGGFeatures(pretrained=not args.no_pretrained_vgg).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    out_dir = Path(args.out_dir) / args.style.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_photos = torch.stack([dataset[i][0] for i in range(min(8, len(dataset)))]).to(device)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"animegan-{args.style.lower()}"):
        mlflow.log_params({**vars(args), "model": "AnimeGenerator", "device": str(device)})
        step = 0
        for epoch in range(args.epochs):
            init_phase = epoch < args.init_epochs
            sums = {"g_total": 0.0, "g_adv": 0.0, "content": 0.0, "gram": 0.0, "d_total": 0.0}
            batches = 0
            for photo, anime, anime_gray, smooth_gray in loader:
                photo, anime = photo.to(device), anime.to(device)
                anime_gray, smooth_gray = anime_gray.to(device), smooth_gray.to(device)
                fake = generator(photo)
                content = F.l1_loss(vgg(fake), vgg(photo))

                if init_phase:
                    g_loss = args.w_con * content
                    g_adv = gram_loss = d_loss = torch.zeros((), device=device)
                else:
                    # --- Discriminateur : vrai = anime ; faux = généré, anime gris, anime lissé
                    d_loss = (
                        (discriminator(anime) - 1.0).pow(2).mean()
                        + discriminator(fake.detach()).pow(2).mean()
                        + discriminator(anime_gray).pow(2).mean()
                        + 0.5 * discriminator(smooth_gray).pow(2).mean()
                    )
                    opt_d.zero_grad()
                    d_loss.backward()
                    opt_d.step()

                    # --- Générateur
                    g_adv = (discriminator(fake) - 1.0).pow(2).mean()
                    gram_loss = F.l1_loss(gram(vgg(to_gray3(fake))), gram(vgg(anime_gray)))
                    g_loss = (
                        args.w_adv * g_adv
                        + args.w_con * content
                        + args.w_gram * gram_loss
                        + args.w_col * color_loss(fake, photo)
                    )

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                values = (g_loss, g_adv, content, gram_loss, d_loss)
                for key, value in zip(sums, values):
                    sums[key] += float(value)
                batches += 1
                step += 1
                if args.max_steps and batches >= args.max_steps:
                    break

            mlflow.log_metrics({k: v / max(batches, 1) for k, v in sums.items()}, step=epoch)
            generator.eval()
            with torch.no_grad():
                grid = comparison_grid(fixed_photos, generator(fixed_photos))
            generator.train()
            mlflow.log_image(grid, f"samples/epoch_{epoch:03d}.png")
            torch.save(generator.state_dict(), out_dir / "generator.pt")
            phase = "init" if init_phase else "adverse"
            print(f"époque {epoch + 1}/{args.epochs} ({phase}) g={sums['g_total'] / batches:.4f}")

        mlflow.log_artifact(str(out_dir / "generator.pt"), artifact_path="checkpoints")


if __name__ == "__main__":
    main()
