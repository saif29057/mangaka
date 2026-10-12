"""Entraînement de la baseline CycleGAN (128x128, style Hayao) avec suivi MLflow.

Exemple :
    python -m src.data.dataset --style Hayao
    python -m src.models.train_cyclegan --style Hayao --epochs 20

Principe (Zhu et al., 2017) : deux générateurs et deux discriminateurs.
  - G_pa : photo -> anime        D_a : vrai / faux pour les images d'anime
  - G_ap : anime -> photo        D_p : vrai / faux pour les photos
Pertes du générateur :
  - adverse (LSGAN)  : G_pa(photo) doit tromper D_a, G_ap(anime) doit tromper D_p ;
  - cycle (L1)       : G_ap(G_pa(photo)) ≈ photo et G_pa(G_ap(anime)) ≈ anime,
                       c'est ce qui remplace les paires d'images absentes du dataset ;
  - identité (L1)    : G_pa(anime) ≈ anime, limite les changements de couleurs inutiles.
Seul G_pa est exporté ensuite (c'est lui qui sert à l'inférence).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import AnimeDataset
from src.models.architecture import PatchDiscriminator, ResnetGenerator
from src.utils.visualization import comparison_grid


class ImagePool:
    """Historique de 50 images générées (astuce de CycleGAN).

    Le discriminateur voit aussi d'anciennes images générées, pas seulement celles du
    générateur actuel : cela évite que les deux réseaux oscillent.
    """

    def __init__(self, size: int = 50):
        self.size = size
        self.images: list[torch.Tensor] = []

    def query(self, batch: torch.Tensor) -> torch.Tensor:
        output = []
        for image in batch.detach():
            if len(self.images) < self.size:
                self.images.append(image)
                output.append(image)
            elif random.random() < 0.5:
                index = random.randrange(self.size)
                output.append(self.images[index])
                self.images[index] = image
            else:
                output.append(image)
        return torch.stack(output)


def lsgan(prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    """Perte LSGAN : (D(x) - 1)² pour le vrai, D(x)² pour le faux."""
    target = 1.0 if target_is_real else 0.0
    return F.mse_loss(prediction, torch.full_like(prediction, target))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement de la baseline CycleGAN")
    parser.add_argument("--style", default="Hayao", help="Hayao (baseline du projet)")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--decay-epoch", type=int, default=10, help="Début de la baisse du lr")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--w-cycle", type=float, default=10.0)
    parser.add_argument("--w-identity", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0, help="Limite par époque (debug)")
    parser.add_argument("--experiment", default="anime-style-transfer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AnimeDataset(args.raw_dir, args.processed_dir, args.style, args.size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )

    g_pa, g_ap = ResnetGenerator().to(device), ResnetGenerator().to(device)
    d_a, d_p = PatchDiscriminator().to(device), PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam(
        list(g_pa.parameters()) + list(g_ap.parameters()), lr=args.lr, betas=(0.5, 0.999)
    )
    opt_d = torch.optim.Adam(
        list(d_a.parameters()) + list(d_p.parameters()), lr=args.lr, betas=(0.5, 0.999)
    )

    # lr constant jusqu'à decay_epoch, puis décroissance linéaire vers 0 (comme l'article).
    def lr_factor(epoch: int) -> float:
        remaining = max(1, args.epochs - args.decay_epoch)
        return 1.0 - max(0, epoch - args.decay_epoch) / remaining

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_factor) for opt in (opt_g, opt_d)]
    pool_anime, pool_photo = ImagePool(), ImagePool()

    out_dir = Path(args.out_dir) / f"cyclegan-{args.style.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_photos = torch.stack([dataset[i][0] for i in range(min(8, len(dataset)))]).to(device)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"cyclegan-{args.style.lower()}"):
        mlflow.log_params(
            {**vars(args), "model": "CycleGAN-ResnetGenerator", "device": str(device)}
        )
        for epoch in range(args.epochs):
            sums = {"g_total": 0.0, "g_adv": 0.0, "cycle": 0.0, "identity": 0.0, "d_total": 0.0}
            batches = 0
            for photo, anime, _, _ in loader:
                photo, anime = photo.to(device), anime.to(device)

                # --- Générateurs (les discriminateurs sont figés pendant cette étape)
                fake_anime, fake_photo = g_pa(photo), g_ap(anime)
                g_adv = lsgan(d_a(fake_anime), True) + lsgan(d_p(fake_photo), True)
                cycle = F.l1_loss(g_ap(fake_anime), photo) + F.l1_loss(g_pa(fake_photo), anime)
                identity = F.l1_loss(g_pa(anime), anime) + F.l1_loss(g_ap(photo), photo)
                g_loss = g_adv + args.w_cycle * cycle + args.w_identity * identity
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                # --- Discriminateurs : vrai = image du dataset, faux = image de l'historique
                d_loss = 0.5 * (
                    lsgan(d_a(anime), True)
                    + lsgan(d_a(pool_anime.query(fake_anime)), False)
                    + lsgan(d_p(photo), True)
                    + lsgan(d_p(pool_photo.query(fake_photo)), False)
                )
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                values = (g_loss, g_adv, cycle, identity, d_loss)
                for key, value in zip(sums, values):
                    sums[key] += value.item()
                batches += 1
                if args.max_steps and batches >= args.max_steps:
                    break

            for scheduler in schedulers:
                scheduler.step()
            metrics = {k: v / max(batches, 1) for k, v in sums.items()}
            metrics["lr"] = opt_g.param_groups[0]["lr"]
            mlflow.log_metrics(metrics, step=epoch)
            g_pa.eval()
            with torch.no_grad():
                grid = comparison_grid(fixed_photos, g_pa(fixed_photos))
            g_pa.train()
            mlflow.log_image(grid, f"samples/epoch_{epoch:03d}.png")
            torch.save(g_pa.state_dict(), out_dir / "generator.pt")
            print(
                f"époque {epoch + 1}/{args.epochs} g={metrics['g_total']:.4f} "
                f"d={metrics['d_total']:.4f}"
            )

        mlflow.log_artifact(str(out_dir / "generator.pt"), artifact_path="checkpoints")


if __name__ == "__main__":
    main()
