"""
Export a human-review pack from geometric_stack_dataset.json.

For each selected task:
- Save composite image (.png if Pillow exists, else .ppm)
- Save easy-to-read .txt with desired order and piece details
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


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


def write_image(path_base: Path, image_rows: List[str], scale: int = 16) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export image + human-readable order review pack.")
    parser.add_argument("--dataset", default="geometric_stack_bench/geometric_stack_dataset.json")
    parser.add_argument("--outdir", default="geometric_stack_bench/review_pack")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    sel = rows[args.offset : args.offset + args.count]

    index_lines = []
    for i, row in enumerate(sel, start=args.offset):
        tid = row.get("id", f"sample_{i:05d}")
        base = outdir / f"{i:05d}_{tid}"

        image_file = write_image(base, row["final_image"], scale=16)

        true_order = row.get("true_order", [])
        answer_layers = row.get("answer_layers", [])
        pieces = row.get("pieces", [])
        order_human = " -> ".join(true_order)

        txt = []
        txt.append(f"Task ID: {tid}")
        txt.append(f"Difficulty: {row.get('difficulty')}")
        txt.append(f"Canvas: {row.get('width')}x{row.get('height')}")
        txt.append(f"Valid order count: {row.get('valid_order_count')}")
        txt.append("")
        txt.append("Desired placement order (BOTTOM -> TOP):")
        txt.append(order_human if order_human else "(not available)")
        if answer_layers:
            txt.append("")
            txt.append("Structured answer layers (z: bottom -> top):")
            for layer in sorted(answer_layers, key=lambda d: int(d.get("z", 0))):
                txt.append(
                    f"- z={layer.get('z')}: {layer.get('piece_id')} "
                    f"shape={layer.get('shape')}, x={layer.get('x')}, y={layer.get('y')}"
                )
        txt.append("")
        txt.append("Pieces:")
        for p in pieces:
            txt.append(
                f"- {p['piece_id']}: shape={p['shape']}, color={p['color']}, "
                f"x={p['x']}, y={p['y']}, size={p['size']}"
            )

        txt_path = base.with_suffix(".txt")
        txt_path.write_text("\n".join(txt), encoding="utf-8")

        index_lines.append(f"{i:05d} | {tid} | {image_file} | {txt_path.name}")

    (outdir / "INDEX.txt").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"Exported {len(sel)} tasks to {outdir}")
    print(f"Index file: {outdir / 'INDEX.txt'}")


if __name__ == "__main__":
    main()
