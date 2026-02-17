"""
Save hand-crafted geometric stack sample images for qualitative review.

Outputs:
- composite image files (.png if Pillow exists, else .ppm)
- a metadata.json with pieces, true order, and ambiguity counts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from geometric_stack_bench.geometric_stack_env import (
    GeometricStackEnvironment,
    Piece,
    ShapeType,
)


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


def _write_image(path: Path, image_rows: List[str], scale: int = 16) -> str:
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
        out = path.with_suffix(".png")
        img.save(out)
        return out.name
    except Exception:
        # Portable fallback: write PPM
        out = path.with_suffix(".ppm")
        with out.open("w", encoding="utf-8") as f:
            f.write(f"P3\n{w * scale} {h * scale}\n255\n")
            for y in range(h):
                row_rgb = []
                for x in range(w):
                    rgb = COLOR_RGB.get(image_rows[y][x], (0, 0, 0))
                    row_rgb.extend([f"{rgb[0]} {rgb[1]} {rgb[2]}"] * scale)
                expanded_line = " ".join(row_rgb)
                for _ in range(scale):
                    f.write(expanded_line + "\n")
        return out.name


def _piece_dict(p: Piece) -> Dict:
    return {
        "piece_id": p.piece_id,
        "shape": p.shape.value,
        "color": p.color,
        "x": p.x,
        "y": p.y,
        "size": p.size,
    }


def _scenes() -> List[Tuple[str, int, int, List[Piece], List[str], str]]:
    return [
        (
            "same_shape_offset_chain",
            16,
            16,
            [
                Piece("P1", ShapeType.CIRCLE, "R", 3, 3, 10),
                Piece("P2", ShapeType.CIRCLE, "G", 5, 4, 8),
                Piece("P3", ShapeType.CIRCLE, "B", 7, 5, 6),
                Piece("P4", ShapeType.CIRCLE, "Y", 8, 6, 4),
            ],
            ["P1", "P2", "P3", "P4"],
            "Same-shape chain with subtle offsets; tests fine occlusion reasoning.",
        ),
        (
            "symmetry_trap_triangle_over_circle",
            16,
            16,
            [
                Piece("P1", ShapeType.SQUARE, "C", 2, 2, 11),
                Piece("P2", ShapeType.CIRCLE, "M", 4, 4, 8),
                Piece("P3", ShapeType.TRIANGLE, "Y", 6, 5, 6),
                Piece("P4", ShapeType.SQUARE, "B", 10, 2, 4),
            ],
            ["P1", "P4", "P2", "P3"],
            "Symmetry-like overlaps with one distinguishing boundary segment.",
        ),
        (
            "decoy_visibility_with_distractor",
            18,
            18,
            [
                Piece("P1", ShapeType.SQUARE, "R", 2, 2, 12),
                Piece("P2", ShapeType.TRIANGLE, "G", 5, 5, 9),
                Piece("P3", ShapeType.CIRCLE, "B", 8, 8, 6),
                Piece("P4", ShapeType.SQUARE, "Y", 13, 1, 4),  # distractor, low overlap
                Piece("P5", ShapeType.TRIANGLE, "M", 10, 11, 5),  # side distractor
            ],
            ["P1", "P4", "P2", "P5", "P3"],
            "Main stack plus side distractors; breaks naive largest-first heuristic.",
        ),
        (
            "near_collision_pixel_gap",
            16,
            16,
            [
                Piece("P1", ShapeType.SQUARE, "G", 1, 4, 10),
                Piece("P2", ShapeType.SQUARE, "R", 10, 4, 5),  # near-touch right side
                Piece("P3", ShapeType.CIRCLE, "B", 7, 1, 6),
                Piece("P4", ShapeType.TRIANGLE, "Y", 6, 7, 7),
            ],
            ["P1", "P2", "P3", "P4"],
            "Near-collision geometry (small gap/contact regions) for exact spatial checks.",
        ),
        (
            "partial_fragment_reveal",
            18,
            18,
            [
                Piece("P1", ShapeType.SQUARE, "C", 3, 3, 12),
                Piece("P2", ShapeType.SQUARE, "M", 5, 5, 10),
                Piece("P3", ShapeType.TRIANGLE, "Y", 7, 7, 8),
                Piece("P4", ShapeType.CIRCLE, "R", 9, 8, 6),
                Piece("P5", ShapeType.CIRCLE, "B", 11, 9, 4),
            ],
            ["P1", "P2", "P3", "P4", "P5"],
            "Deep stack where lower layers are only partially visible as thin fragments.",
        ),
    ]


def main() -> None:
    env = GeometricStackEnvironment(seed=42)
    out_dir = Path("geometric_stack_bench/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = []
    for name, width, height, pieces, order, note in _scenes():
        pieces_by_id = {p.piece_id: p for p in pieces}
        final_image = env.render_order(pieces_by_id, order, width, height)
        valid_count = env._count_valid_orders(pieces, final_image, limit=100000)
        filename = _write_image(out_dir / name, final_image, scale=18)

        meta.append(
            {
                "name": name,
                "file": filename,
                "width": width,
                "height": height,
                "note": note,
                "true_order_bottom_to_top": order,
                "valid_order_count": valid_count,
                "pieces": [_piece_dict(p) for p in pieces],
            }
        )

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(meta)} sample scenes to: {out_dir}")
    for row in meta:
        print(f"- {row['file']} | valid_order_count={row['valid_order_count']} | {row['name']}")


if __name__ == "__main__":
    main()
