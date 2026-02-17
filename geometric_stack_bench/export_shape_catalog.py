"""
Export a catalog of canonical single-shape images for shape sizes found in a dataset.

Each output image contains exactly one piece centered on the canvas.
Only sizes that actually appear in the dataset are exported.

Usage:
  python geometric_stack_bench/export_shape_catalog.py \
    --dataset geometric_stack_bench/geometric_stack_dataset_review_samples.json \
    --outdir geometric_stack_bench/shape_catalog_review_samples
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from geometric_stack_bench.geometric_stack_env import GeometricStackEnvironment, Piece, ShapeType


COLOR_RGB = {
    ".": (245, 245, 245),
    "R": (220, 50, 47),
    "G": (38, 139, 34),
    "B": (38, 93, 220),
    "Y": (210, 170, 20),
    "C": (30, 170, 170),
    "M": (170, 30, 170),
    "W": (40, 40, 40),
    "K": (10, 10, 10),
}

SHAPE_COLOR = {
    ShapeType.SQUARE: "R",
    ShapeType.CIRCLE: "G",
    ShapeType.TRIANGLE: "B",
}


def write_image(path_base: Path, image_rows: List[str], scale: int = 20) -> str:
    h = len(image_rows)
    w = len(image_rows[0])
    try:
        from PIL import Image

        img = Image.new("RGB", (w * scale, h * scale), COLOR_RGB["."])
        px = img.load()
        for y in range(h):
            for x in range(w):
                rgb = COLOR_RGB.get(image_rows[y][x], (0, 0, 0))
                for yy in range(y * scale, (y + 1) * scale):
                    for xx in range(x * scale, (x + 1) * scale):
                        px[xx, yy] = rgb
        out = path_base.with_suffix(".png")
        img.save(out)
        return out.name
    except Exception:
        out = path_base.with_suffix(".ppm")
        with out.open("w", encoding="utf-8") as f:
            f.write(f"P3\n{w * scale} {h * scale}\n255\n")
            for y in range(h):
                row_rgb = []
                for x in range(w):
                    rgb = COLOR_RGB.get(image_rows[y][x], (0, 0, 0))
                    row_rgb.extend([f"{rgb[0]} {rgb[1]} {rgb[2]}"] * scale)
                line = " ".join(row_rgb)
                for _ in range(scale):
                    f.write(line + "\n")
        return out.name


def _axis_coords(anchor_x: float, anchor_y: float, width: int, height: int) -> Tuple[float, float]:
    return anchor_x - width / 2.0, height / 2.0 - anchor_y


def main() -> None:
    parser = argparse.ArgumentParser(description="Export single-shape reference catalog from dataset sizes.")
    parser.add_argument("--dataset", default="geometric_stack_bench/geometric_stack_dataset_review_samples.json")
    parser.add_argument("--outdir", default="geometric_stack_bench/shape_catalog_review_samples")
    parser.add_argument("--scale", type=int, default=20)
    args = parser.parse_args()

    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not rows:
        raise RuntimeError("Dataset is empty.")

    shape_sizes: Dict[str, set] = defaultdict(set)
    max_canvas = 0
    for row in rows:
        max_canvas = max(max_canvas, int(row.get("width", 0)), int(row.get("height", 0)))
        for p in row.get("pieces", []):
            shape_sizes[str(p["shape"])].add(int(p["size"]))

    if max_canvas <= 0:
        max_canvas = 16

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = GeometricStackEnvironment(seed=0, canvas_size=max_canvas)
    manifest = {
        "dataset": str(args.dataset),
        "canvas_size": max_canvas,
        "shape_sizes": {k: sorted(v) for k, v in shape_sizes.items()},
        "files": [],
    }

    for shape in [ShapeType.SQUARE, ShapeType.CIRCLE, ShapeType.TRIANGLE]:
        sizes = sorted(shape_sizes.get(shape.value, set()))
        for size in sizes:
            x = (max_canvas - size) // 2
            y = (max_canvas - size) // 2
            piece = Piece(
                piece_id="P1",
                shape=shape,
                color=SHAPE_COLOR[shape],
                x=x,
                y=y,
                size=size,
            )
            image = env.render_order({"P1": piece}, ["P1"], max_canvas, max_canvas)
            stem = f"{shape.value}_size_{size:02d}"
            image_name = write_image(outdir / stem, image, scale=max(args.scale, 1))

            axis_x, axis_y = _axis_coords(piece.anchor_x, piece.anchor_y, max_canvas, max_canvas)
            txt_lines = [
                f"shape={shape.value}",
                f"size={size}",
                f"canvas={max_canvas}x{max_canvas}",
                f"x={piece.x}",
                f"y={piece.y}",
                f"anchor=({piece.anchor_x:.1f},{piece.anchor_y:.1f})",
                f"axis=({axis_x:.1f},{axis_y:.1f})",
                f"image={image_name}",
            ]
            (outdir / f"{stem}.txt").write_text("\n".join(txt_lines), encoding="utf-8")

            manifest["files"].append(
                {
                    "shape": shape.value,
                    "size": size,
                    "image": image_name,
                    "meta": f"{stem}.txt",
                }
            )

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved shape catalog to: {outdir}")
    print(f"Total images: {len(manifest['files'])}")
    for shape in ["square", "circle", "triangle"]:
        print(f"{shape}: sizes={sorted(shape_sizes.get(shape, set()))}")


if __name__ == "__main__":
    main()
