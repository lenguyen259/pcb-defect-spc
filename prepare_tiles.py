"""Cut PKU-Market-PCB full boards (~3000x1600 px, defects ~70 px) into 640 px YOLO tiles.

Board-disjoint split (ImageSets/train_board.txt, val_board.txt) so no board appears in both sets.
Usage: python prepare_tiles.py <pcb_root> <out_root> [neg_ratio]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2

TILE = 640


def positions(length: int) -> list[int]:
    if length <= TILE:
        return [0]
    n = -(-length // TILE)
    step = (length - TILE) / (n - 1)
    return [round(i * step) for i in range(n)]


def read_split(root: Path, name: str) -> list[str]:
    return [Path(x.strip()).stem for x in (root / "ImageSets" / name).read_text().split() if x.strip()]


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    neg_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    rng = random.Random(0)
    stats = {}
    for split, listing in (("train", "train_board.txt"), ("val", "val_board.txt")):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        pos = neg = 0
        for stem in read_split(root, listing):
            img = cv2.imread(str(root / "images" / f"{stem}.jpg"))
            H, W = img.shape[:2]
            boxes = []
            for line in (root / "labels" / f"{stem}.txt").read_text().splitlines():
                c, cx, cy, w, h = line.split()
                boxes.append((int(c), float(cx) * W, float(cy) * H, float(w) * W, float(h) * H))
            for y0 in positions(H):
                for x0 in positions(W):
                    rows = []
                    for c, cx, cy, w, h in boxes:
                        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                        ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0 + TILE), min(y2, y0 + TILE)
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        if (ix2 - ix1) * (iy2 - iy1) < 0.6 * w * h:
                            continue
                        rows.append(f"{c} {((ix1+ix2)/2-x0)/TILE:.6f} {((iy1+iy2)/2-y0)/TILE:.6f} "
                                    f"{(ix2-ix1)/TILE:.6f} {(iy2-iy1)/TILE:.6f}")
                    if not rows:
                        if split == "train" and rng.random() > neg_ratio:
                            continue
                        neg += 1
                    else:
                        pos += 1
                    name = f"{stem}_{x0}_{y0}"
                    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img[y0:y0 + TILE, x0:x0 + TILE],
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (out / "labels" / split / f"{name}.txt").write_text("\n".join(rows))
        stats[split] = (pos, neg)
    names = (root / "classes.txt").read_text().split()
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnc: {len(names)}\nnames: {names}\n")
    print("tiles (with defects, background):", stats)


if __name__ == "__main__":
    main()
