"""Range l'archive AnimeGAN décompressée dans l'arborescence attendue (voir data/README.md).

    python -m src.data.prepare_raw --src <dossier décompressé> --dst data/raw

L'archive officielle n'a pas toujours la même forme (dossier racine `dataset/`, photos de
test dans `test/test_photo`...). On cherche donc chaque dossier par son nom, puis on vérifie
qu'aucune photo de test n'est aussi une photo d'entraînement (comparaison du contenu).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from src.data.dataset import list_images


def find_dir(root: Path, *candidates: str) -> Path | None:
    """Premier dossier contenant des images dont le chemin se termine par un des candidats."""
    for candidate in candidates:
        parts = tuple(candidate.split("/"))
        for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
            if folder.parts[-len(parts) :] == parts and list_images(folder):
                return folder
    return None


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def copy_images(src: Path, dst: Path, skip: set[str] | None = None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in list_images(src):
        if skip and file_hash(path) in skip:
            continue
        shutil.copy2(path, dst / path.name)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prépare data/raw depuis l'archive AnimeGAN")
    parser.add_argument("--src", required=True, help="dossier où l'archive a été décompressée")
    parser.add_argument("--dst", default="data/raw")
    parser.add_argument("--styles", nargs="+", default=["Hayao", "Shinkai"])
    args = parser.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    train = find_dir(src, "train_photo")
    test = find_dir(src, "test_photo", "test/real", "test")
    if train is None or test is None:
        raise SystemExit(f"train_photo ou test_photo introuvable dans {src}")

    train_hashes = {file_hash(p) for p in list_images(train)}
    n_train = copy_images(train, dst / "train_photo")
    n_test_all = len(list_images(test))
    n_test = copy_images(test, dst / "test_photo", skip=train_hashes)
    print(f"train_photo : {n_train} images  ({train})")
    print(f"test_photo  : {n_test} images  ({test})")
    if n_test != n_test_all:
        print(f"  {n_test_all - n_test} photo(s) de test identiques à l'entraînement : écartées")

    for style in args.styles:
        style_dir = find_dir(src, f"{style}/style")
        if style_dir is None:
            print(f"{style} : dossier style introuvable, ignoré")
            continue
        print(f"{style}/style : {copy_images(style_dir, dst / style / 'style')} images")


if __name__ == "__main__":
    main()
