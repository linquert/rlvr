"""
Geometric Stack RLVR environment.

Task idea:
- Given a set of 2D geometric pieces (shape + color + position + size)
- And a final overlapped composite image
- Predict the placement order of pieces (bottom -> top).

This is fully code-verifiable:
- Rendering is deterministic.
- Candidate orders are verified by exact image reconstruction.
- Reward is variable (exact + partial).
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ShapeType(Enum):
    SQUARE = "square"
    CIRCLE = "circle"
    TRIANGLE = "triangle"


@dataclass(frozen=True)
class Piece:
    piece_id: str
    shape: ShapeType
    color: str
    x: int
    y: int
    size: int

    @property
    def anchor_x(self) -> float:
        # Cell-center anchored coordinate in canvas space.
        return self.x + self.size / 2.0

    @property
    def anchor_y(self) -> float:
        # Cell-center anchored coordinate in canvas space.
        return self.y + self.size / 2.0


@dataclass
class GeometricStackTask:
    width: int
    height: int
    pieces: List[Piece]
    true_order: List[str]  # bottom -> top
    final_image: List[str]  # list of rows with color chars, "." is empty
    difficulty: str
    valid_order_count: int
    valid_orders: Optional[List[List[str]]] = None


class GeometricStackEnvironment:
    def __init__(self, seed: Optional[int] = None, canvas_size: int = 16):
        if seed is not None:
            random.seed(seed)
        self.shape_pool = [ShapeType.SQUARE, ShapeType.CIRCLE, ShapeType.TRIANGLE]
        self.color_pool = list("RGBYCMWK")  # color symbols
        # Standardized square canvas across all difficulties.
        self.canvas_size = canvas_size

    # =========================
    # Rendering
    # =========================
    def _piece_mask(self, piece: Piece, width: int, height: int) -> List[Tuple[int, int]]:
        pixels: List[Tuple[int, int]] = []
        x0, y0, s = piece.x, piece.y, piece.size

        def _point_in_triangle(px: float, py: float, ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
            # Barycentric sign method.
            def sign(x1, y1, x2, y2, x3, y3):
                return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

            b1 = sign(px, py, ax, ay, bx, by) < 0.0
            b2 = sign(px, py, bx, by, cx, cy) < 0.0
            b3 = sign(px, py, cx, cy, ax, ay) < 0.0
            return (b1 == b2) and (b2 == b3)

        for yy in range(height):
            for xx in range(width):
                inside = False
                if piece.shape == ShapeType.SQUARE:
                    inside = x0 <= xx < x0 + s and y0 <= yy < y0 + s
                elif piece.shape == ShapeType.CIRCLE:
                    # Canonical circle anchored to 0.5-grid centers.
                    cx = x0 + s / 2.0
                    cy = y0 + s / 2.0
                    # Use inscribed radius to avoid tiny circles degenerating into full squares
                    # (notably size=3), while preserving recognizable circular edges.
                    r = (s - 1) / 2.0
                    px = xx + 0.5
                    py = yy + 0.5
                    inside = (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2
                elif piece.shape == ShapeType.TRIANGLE:
                    # Canonical upright isosceles triangle anchored to 0.5-grid centers.
                    ax, ay = x0 + s / 2.0, y0
                    bx, by = x0, y0 + s
                    cx2, cy2 = x0 + s, y0 + s
                    px = xx + 0.5
                    py = yy + 0.5
                    inside = _point_in_triangle(px, py, ax, ay, bx, by, cx2, cy2)

                if inside:
                    pixels.append((xx, yy))
        return pixels

    def render_order(
        self,
        pieces_by_id: Dict[str, Piece],
        order_bottom_to_top: Sequence[str],
        width: int,
        height: int,
    ) -> List[str]:
        canvas = [["." for _ in range(width)] for _ in range(height)]
        for pid in order_bottom_to_top:
            p = pieces_by_id[pid]
            for xx, yy in self._piece_mask(p, width, height):
                canvas[yy][xx] = p.color
        return ["".join(row) for row in canvas]

    def _render_order_with_owner(
        self,
        pieces_by_id: Dict[str, Piece],
        order_bottom_to_top: Sequence[str],
        width: int,
        height: int,
    ) -> Tuple[List[str], List[List[Optional[str]]]]:
        canvas = [["." for _ in range(width)] for _ in range(height)]
        owner: List[List[Optional[str]]] = [[None for _ in range(width)] for _ in range(height)]
        for pid in order_bottom_to_top:
            p = pieces_by_id[pid]
            for xx, yy in self._piece_mask(p, width, height):
                canvas[yy][xx] = p.color
                owner[yy][xx] = pid
        return ["".join(row) for row in canvas], owner

    # =========================
    # Task generation
    # =========================
    def _difficulty_cfg(self, difficulty: str) -> Tuple[int, int, int]:
        # num_pieces, canvas_size, max_size
        if difficulty == "easy":
            return 3, self.canvas_size, 5
        if difficulty == "medium":
            return 4, self.canvas_size, 7
        if difficulty == "hard":
            return 5, self.canvas_size, 9
        if difficulty == "expert":
            return 6, self.canvas_size, 11
        raise ValueError(f"Unknown difficulty: {difficulty}")

    def _random_piece(self, idx: int, width: int, height: int, max_size: int, used_colors: set) -> Piece:
        color_choices = [c for c in self.color_pool if c not in used_colors]
        if not color_choices:
            color_choices = self.color_pool[:]  # fallback
        color = random.choice(color_choices)
        used_colors.add(color)
        # Use odd sizes so center anchors are always at .5 precision.
        size_choices = [s for s in range(3, max_size + 1) if s % 2 == 1]
        if not size_choices:
            size_choices = [3]
        size = random.choice(size_choices)
        x = random.randint(0, max(0, width - size))
        y = random.randint(0, max(0, height - size))
        shape = random.choice(self.shape_pool)
        return Piece(piece_id=f"P{idx+1}", shape=shape, color=color, x=x, y=y, size=size)

    def _triangle_has_signature_visibility(
        self,
        piece: Piece,
        owner_map: Sequence[Sequence[Optional[str]]],
        piece_mask: Optional[set] = None,
    ) -> bool:
        """
        Ensure at least one clearly triangle-specific cue is visible:
        - apex visibility near the top tip
        - at least one visible slanted side boundary pixel.
        """
        if piece.shape != ShapeType.TRIANGLE:
            return True

        height = len(owner_map)
        width = len(owner_map[0]) if height > 0 else 0
        if piece_mask is None:
            piece_mask = set(self._piece_mask(piece, width, height))

        apex_x = int(piece.x + piece.size / 2.0 - 0.5)
        apex_y = piece.y
        apex_visible = (
            0 <= apex_x < width
            and 0 <= apex_y < height
            and owner_map[apex_y][apex_x] == piece.piece_id
        )

        left_side_visible = False
        right_side_visible = False
        for yy in range(piece.y, piece.y + piece.size):
            row_pixels = [xx for xx in range(piece.x, piece.x + piece.size) if (xx, yy) in piece_mask]
            if not row_pixels:
                continue
            left_x = min(row_pixels)
            right_x = max(row_pixels)
            if 0 <= yy < height and owner_map[yy][left_x] == piece.piece_id:
                left_side_visible = True
            if 0 <= yy < height and owner_map[yy][right_x] == piece.piece_id:
                right_side_visible = True

        return apex_visible and (left_side_visible or right_side_visible)

    def _passes_visibility_constraints(
        self,
        pieces: Sequence[Piece],
        order: Sequence[str],
        width: int,
        height: int,
        min_visible_pixels_per_piece: int,
        require_triangle_signature: bool,
    ) -> bool:
        pieces_by_id = {p.piece_id: p for p in pieces}
        _img, owner = self._render_order_with_owner(pieces_by_id, order, width, height)

        visible_count: Dict[str, int] = {p.piece_id: 0 for p in pieces}
        for y in range(height):
            for x in range(width):
                pid = owner[y][x]
                if pid is not None:
                    visible_count[pid] += 1

        if any(c < min_visible_pixels_per_piece for c in visible_count.values()):
            return False

        if require_triangle_signature:
            for p in pieces:
                if p.shape == ShapeType.TRIANGLE:
                    mask = set(self._piece_mask(p, width, height))
                    if not self._triangle_has_signature_visibility(p, owner, piece_mask=mask):
                        return False

        return True

    def _count_valid_orders(
        self,
        pieces: List[Piece],
        final_image: List[str],
        limit: int = 100000,
    ) -> int:
        pieces_by_id = {p.piece_id: p for p in pieces}
        ids = [p.piece_id for p in pieces]
        count = 0
        for perm in itertools.permutations(ids):
            rendered = self.render_order(pieces_by_id, perm, len(final_image[0]), len(final_image))
            if rendered == final_image:
                count += 1
                if count >= limit:
                    return count
        return count

    def _enumerate_valid_orders(
        self,
        pieces: List[Piece],
        final_image: List[str],
        max_collect: int = 100000,
    ) -> List[List[str]]:
        pieces_by_id = {p.piece_id: p for p in pieces}
        ids = [p.piece_id for p in pieces]
        out: List[List[str]] = []
        for perm in itertools.permutations(ids):
            rendered = self.render_order(pieces_by_id, perm, len(final_image[0]), len(final_image))
            if rendered == final_image:
                out.append(list(perm))
                if len(out) >= max_collect:
                    break
        return out

    def generate_task(
        self,
        difficulty: str = "medium",
        max_generation_tries: int = 200,
        require_unique_order: bool = True,
        min_overlap_pixels: int = 0,
        min_visible_pixels_per_piece: int = 3,
        require_triangle_signature: bool = True,
        min_valid_order_count: int = 1,
        max_valid_order_count: Optional[int] = None,
    ) -> GeometricStackTask:
        num_pieces, canvas_size, max_size = self._difficulty_cfg(difficulty)
        width = canvas_size
        height = canvas_size

        for _ in range(max_generation_tries):
            used_colors = set()
            pieces = [self._random_piece(i, width, height, max_size, used_colors) for i in range(num_pieces)]
            order = [p.piece_id for p in pieces]
            random.shuffle(order)
            pieces_by_id = {p.piece_id: p for p in pieces}
            final_image = self.render_order(pieces_by_id, order, width, height)
            overlap_pixels = self._compute_overlap_pixels(pieces, width, height)
            if overlap_pixels < min_overlap_pixels:
                continue
            if not self._passes_visibility_constraints(
                pieces,
                order,
                width,
                height,
                min_visible_pixels_per_piece=min_visible_pixels_per_piece,
                require_triangle_signature=require_triangle_signature,
            ):
                continue

            valid_order_count = self._count_valid_orders(
                pieces, final_image, limit=2 if require_unique_order else 100000
            )
            if valid_order_count < min_valid_order_count:
                continue
            if max_valid_order_count is not None and valid_order_count > max_valid_order_count:
                continue
            if require_unique_order:
                if valid_order_count == 1:
                    return GeometricStackTask(
                        width=width,
                        height=height,
                        pieces=pieces,
                        true_order=order,
                        final_image=final_image,
                        difficulty=difficulty,
                        valid_order_count=1,
                        valid_orders=[order],
                    )
            else:
                valid_orders = self._enumerate_valid_orders(pieces, final_image, max_collect=100000)
                return GeometricStackTask(
                    width=width,
                    height=height,
                    pieces=pieces,
                    true_order=order,
                    final_image=final_image,
                    difficulty=difficulty,
                    valid_order_count=valid_order_count,
                    valid_orders=valid_orders,
                )

        raise RuntimeError(f"Failed to generate {'unique' if require_unique_order else 'valid'} task in allotted tries.")

    def _compute_overlap_pixels(self, pieces: List[Piece], width: int, height: int) -> int:
        """
        Count pixels covered by at least 2 pieces, as a simple non-triviality signal.
        """
        cover_count = [[0 for _ in range(width)] for _ in range(height)]
        for p in pieces:
            for xx, yy in self._piece_mask(p, width, height):
                cover_count[yy][xx] += 1
        overlap = 0
        for y in range(height):
            for x in range(width):
                if cover_count[y][x] >= 2:
                    overlap += 1
        return overlap

    # =========================
    # Verification + reward
    # =========================
    def _pairwise_against_valid_orders(
        self,
        ids: Sequence[str],
        pred: Sequence[str],
        valid_orders: Sequence[Sequence[str]],
    ) -> float:
        best_pairwise = 0.0
        pred_pos = {pid: i for i, pid in enumerate(pred)}
        for vo in valid_orders:
            true_pos = {pid: i for i, pid in enumerate(vo)}
            pairs = 0
            agrees = 0
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    pairs += 1
                    if (true_pos[a] - true_pos[b]) * (pred_pos[a] - pred_pos[b]) > 0:
                        agrees += 1
            score = agrees / max(pairs, 1)
            if score > best_pairwise:
                best_pairwise = score
        return best_pairwise

    def evaluate_order(self, task: GeometricStackTask, predicted_order: Sequence[str]) -> Dict[str, float]:
        ids = [p.piece_id for p in task.pieces]
        id_set = set(ids)
        pred = list(predicted_order)

        if len(pred) != len(ids) or set(pred) != id_set:
            return {
                "success": 0.0,
                "valid_order_match": 0.0,
                "pixel_accuracy": 0.0,
                "pairwise_accuracy": 0.0,
                "correct_actions": 0.0,
                "action_accuracy": 0.0,
                "reward": -1.0,
            }

        pieces_by_id = {p.piece_id: p for p in task.pieces}
        rendered = self.render_order(pieces_by_id, pred, task.width, task.height)

        total = task.width * task.height
        same = 0
        for y in range(task.height):
            for x in range(task.width):
                if rendered[y][x] == task.final_image[y][x]:
                    same += 1
        pixel_accuracy = same / max(total, 1)

        # Pairwise agreement against the closest valid order (not just one reference order).
        valid_orders = task.valid_orders or self._enumerate_valid_orders(task.pieces, task.final_image, max_collect=100000)
        pairwise_accuracy = self._pairwise_against_valid_orders(ids, pred, valid_orders)

        valid_order_match = 1.0 if pred in valid_orders else 0.0
        success = 1.0 if rendered == task.final_image else 0.0

        # Count how many prefix actions can still be completed into at least one valid solution.
        correct_actions = 0
        for k in range(1, len(pred) + 1):
            prefix = pred[:k]
            if self._prefix_is_extendable(prefix, valid_orders):
                correct_actions = k
            else:
                break
        action_accuracy = correct_actions / max(len(pred), 1)

        # Variable reward:
        # - success by reconstruction is primary
        # - partial credit from actionable progress + image similarity
        reward = 1.0 if success == 1.0 else (0.55 * pixel_accuracy + 0.30 * action_accuracy + 0.15 * pairwise_accuracy)

        return {
            "success": success,
            "valid_order_match": valid_order_match,
            "pixel_accuracy": pixel_accuracy,
            "pairwise_accuracy": pairwise_accuracy,
            "correct_actions": float(correct_actions),
            "action_accuracy": action_accuracy,
            "reward": reward,
        }

    def order_to_layers(self, task: GeometricStackTask, order_bottom_to_top: Sequence[str]) -> List[Dict[str, Any]]:
        pieces_by_id = {p.piece_id: p for p in task.pieces}
        out: List[Dict[str, Any]] = []
        for z, pid in enumerate(order_bottom_to_top):
            p = pieces_by_id[pid]
            out.append(
                {
                    "piece_id": pid,
                    "shape": p.shape.value,
                    "x": p.x,
                    "y": p.y,
                    "z": z,
                }
            )
        return out

    @staticmethod
    def _normalize_layers_to_order(predicted_layers: Sequence[Dict[str, Any]]) -> List[str]:
        # z increases bottom -> top.
        ordered = sorted(
            predicted_layers,
            key=lambda d: (int(d.get("z", 0)), str(d.get("piece_id", ""))),
        )
        return [str(d.get("piece_id", "")) for d in ordered]

    def evaluate_layers(self, task: GeometricStackTask, predicted_layers: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        ids = [p.piece_id for p in task.pieces]
        id_set = set(ids)
        pieces_by_id = {p.piece_id: p for p in task.pieces}
        required = {"piece_id", "shape", "x", "y", "z"}

        if len(predicted_layers) != len(ids):
            return {
                "success": 0.0,
                "strict_success": 0.0,
                "metadata_accuracy": 0.0,
                "z_order_accuracy": 0.0,
                "valid_order_match": 0.0,
                "pixel_accuracy": 0.0,
                "reward": -1.0,
            }

        pred_ids: List[str] = []
        metadata_hits = 0
        for row in predicted_layers:
            if not isinstance(row, dict):
                return {
                    "success": 0.0,
                    "strict_success": 0.0,
                    "metadata_accuracy": 0.0,
                    "z_order_accuracy": 0.0,
                    "valid_order_match": 0.0,
                    "pixel_accuracy": 0.0,
                    "reward": -1.0,
                }
            if not required.issubset(set(row.keys())):
                return {
                    "success": 0.0,
                    "strict_success": 0.0,
                    "metadata_accuracy": 0.0,
                    "z_order_accuracy": 0.0,
                    "valid_order_match": 0.0,
                    "pixel_accuracy": 0.0,
                    "reward": -1.0,
                }
            pid = str(row["piece_id"])
            pred_ids.append(pid)
            p = pieces_by_id.get(pid)
            if p is None:
                continue
            if (
                str(row.get("shape", "")).lower() == p.shape.value
                and int(row.get("x")) == p.x
                and int(row.get("y")) == p.y
            ):
                metadata_hits += 1

        if len(set(pred_ids)) != len(pred_ids) or set(pred_ids) != id_set:
            return {
                "success": 0.0,
                "strict_success": 0.0,
                "metadata_accuracy": 0.0,
                "z_order_accuracy": 0.0,
                "valid_order_match": 0.0,
                "pixel_accuracy": 0.0,
                "reward": -1.0,
            }

        pred_order = self._normalize_layers_to_order(predicted_layers)
        visual_metrics = self.evaluate_order(task, pred_order)
        valid_orders = task.valid_orders or self._enumerate_valid_orders(task.pieces, task.final_image, max_collect=100000)
        z_order_accuracy = self._pairwise_against_valid_orders(ids, pred_order, valid_orders)
        metadata_accuracy = metadata_hits / max(len(ids), 1)
        valid_order_match = 1.0 if pred_order in valid_orders else 0.0
        strict_success = 1.0 if (valid_order_match == 1.0 and metadata_accuracy == 1.0) else 0.0

        if strict_success == 1.0:
            reward = 1.0
        else:
            # Prioritize symbolic construction correctness (z-order + metadata),
            # keep image similarity as bounded fallback.
            reward = min(
                0.95,
                0.70 * z_order_accuracy + 0.20 * metadata_accuracy + 0.10 * visual_metrics["pixel_accuracy"],
            )

        return {
            "success": visual_metrics["success"],
            "strict_success": strict_success,
            "metadata_accuracy": metadata_accuracy,
            "z_order_accuracy": z_order_accuracy,
            "valid_order_match": valid_order_match,
            "pixel_accuracy": visual_metrics["pixel_accuracy"],
            "pairwise_accuracy": visual_metrics["pairwise_accuracy"],
            "action_accuracy": visual_metrics["action_accuracy"],
            "reward": reward,
        }

    @staticmethod
    def _prefix_is_extendable(prefix: Sequence[str], valid_orders: Sequence[Sequence[str]]) -> bool:
        for order in valid_orders:
            if len(prefix) <= len(order) and list(order[: len(prefix)]) == list(prefix):
                return True
        return False

    # =========================
    # Serialization / prompting
    # =========================
    @staticmethod
    def _to_axis_coords(anchor_x: float, anchor_y: float, width: int, height: int) -> Tuple[float, float]:
        # Cartesian-style axis with origin at canvas center, +x right, +y up.
        axis_x = anchor_x - width / 2.0
        axis_y = height / 2.0 - anchor_y
        return axis_x, axis_y

    @staticmethod
    def _format_image_with_xy_labels(image_rows: Sequence[str]) -> str:
        width = len(image_rows[0])
        header = "   " + "".join(str(x % 10) for x in range(width))
        rows = [header]
        for y, row in enumerate(image_rows):
            rows.append(f"{y:02d} {row}")
        return "\n".join(rows)

    def task_to_dict(self, task: GeometricStackTask, include_answer: bool = False) -> Dict:
        pieces = []
        for p in task.pieces:
            axis_x, axis_y = self._to_axis_coords(p.anchor_x, p.anchor_y, task.width, task.height)
            pieces.append(
                {
                    "piece_id": p.piece_id,
                    "shape": p.shape.value,
                    "color": p.color,
                    "x": p.x,
                    "y": p.y,
                    "size": p.size,
                    "anchor_x": round(p.anchor_x, 1),
                    "anchor_y": round(p.anchor_y, 1),
                    "axis_x": round(axis_x, 1),
                    "axis_y": round(axis_y, 1),
                }
            )
        out = {
            "width": task.width,
            "height": task.height,
            "difficulty": task.difficulty,
            "pieces": pieces,
            "final_image": task.final_image,
            "valid_order_count": task.valid_order_count,
        }
        if include_answer:
            out["true_order"] = task.true_order
            out["valid_orders"] = task.valid_orders if task.valid_orders is not None else [task.true_order]
            out["answer_layers"] = self.order_to_layers(task, task.true_order)
        return out

    @staticmethod
    def task_from_dict(row: Dict[str, Any]) -> GeometricStackTask:
        pieces = [
            Piece(
                piece_id=str(p["piece_id"]),
                shape=ShapeType(str(p["shape"])),
                color=str(p["color"]),
                x=int(p["x"]),
                y=int(p["y"]),
                size=int(p["size"]),
            )
            for p in row["pieces"]
        ]
        return GeometricStackTask(
            width=int(row["width"]),
            height=int(row["height"]),
            pieces=pieces,
            true_order=[str(x) for x in row.get("true_order", [])],
            final_image=[str(r) for r in row["final_image"]],
            difficulty=str(row["difficulty"]),
            valid_order_count=int(row["valid_order_count"]),
            valid_orders=row.get("valid_orders"),
        )

    def format_task_for_llm(self, task: GeometricStackTask) -> str:
        piece_lines = []
        for p in task.pieces:
            axis_x, axis_y = self._to_axis_coords(p.anchor_x, p.anchor_y, task.width, task.height)
            piece_lines.append(
                f"- {p.piece_id}: shape={p.shape.value}, color={p.color}, x={p.x}, y={p.y}, size={p.size}, "
                f"anchor=({p.anchor_x:.1f},{p.anchor_y:.1f}), axis=({axis_x:.1f},{axis_y:.1f})"
            )
        image_block = self._format_image_with_xy_labels(task.final_image)
        return f"""Geometric Stack Task (difficulty={task.difficulty})

You are given geometric pieces and the final overlapped 2D composite image.
Infer the placement order from bottom to top.

Coordinate system:
- Grid coordinates use 1 unit intervals per cell.
- Top-left cell is (0,0), x increases right, y increases down.
- Piece anchors are fixed to 0.5 precision only (example valid anchors: (0.5,1.5), (7.5,4.5)).
- Axis coordinates are also provided with origin at canvas center (+y upward).

Pieces:
{chr(10).join(piece_lines)}

Final composite image ({task.width}x{task.height}):
{image_block}

Output format (REQUIRED):
[
  {{"piece_id":"P1","shape":"square","x":4,"y":6,"z":0}},
  {{"piece_id":"P3","shape":"triangle","x":2,"y":5,"z":1}},
  ...
]
Where:
- z = layer index from bottom (0) to top (N-1)
- Provide one entry per piece, no duplicates.

Rules:
1) Use each piece exactly once.
2) Order is bottom -> top.
3) Your goal is exact image reconstruction.
"""


def export_dataset(
    filename: str,
    num_tasks: int = 100,
    difficulty_dist: Optional[Dict[str, float]] = None,
    seed: int = 42,
    require_unique_order: bool = False,
    min_overlap_pixels: Optional[Dict[str, int]] = None,
    min_visible_pixels_per_piece: int = 3,
    require_triangle_signature: bool = True,
    min_valid_orders_nonunique: int = 2,
    max_valid_orders_nonunique: int = 24,
    validate_answers: bool = True,
) -> None:
    env = GeometricStackEnvironment(seed=seed)
    if difficulty_dist is None:
        difficulty_dist = {"easy": 0.25, "medium": 0.35, "hard": 0.25, "expert": 0.15}
    if min_overlap_pixels is None:
        min_overlap_pixels = {"easy": 6, "medium": 12, "hard": 18, "expert": 24}

    difficulties = list(difficulty_dist.keys())
    weights = list(difficulty_dist.values())

    rows = []
    per_difficulty_tries = {
        "easy": 400,
        "medium": 700,
        "hard": 1200,
        "expert": 2000,
    }

    for idx in range(num_tasks):
        diff = random.choices(difficulties, weights=weights)[0]
        max_tries = per_difficulty_tries.get(diff, 1200)
        task = env.generate_task(
            difficulty=diff,
            max_generation_tries=max_tries,
            require_unique_order=require_unique_order,
            min_overlap_pixels=min_overlap_pixels.get(diff, 8),
            min_visible_pixels_per_piece=min_visible_pixels_per_piece,
            require_triangle_signature=require_triangle_signature,
            min_valid_order_count=1 if require_unique_order else min_valid_orders_nonunique,
            max_valid_order_count=1 if require_unique_order else max_valid_orders_nonunique,
        )
        row = env.task_to_dict(task, include_answer=True)
        row["id"] = f"geo_{idx:05d}"
        row["prompt"] = env.format_task_for_llm(task)
        if validate_answers:
            metrics = env.evaluate_layers(task, row["answer_layers"])
            if metrics["strict_success"] != 1.0:
                raise RuntimeError(f"Generated invalid answer_layers for row {row['id']}: {metrics}")
        rows.append(row)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def validate_dataset_file(filename: str) -> Dict[str, Any]:
    env = GeometricStackEnvironment(seed=123)
    with open(filename, "r", encoding="utf-8") as f:
        rows = json.load(f)

    issues: List[str] = []
    for i, row in enumerate(rows):
        tid = row.get("id", f"row_{i:05d}")
        try:
            task = env.task_from_dict(row)
            pieces_by_id = {p.piece_id: p for p in task.pieces}
            if task.true_order:
                rendered = env.render_order(pieces_by_id, task.true_order, task.width, task.height)
                if rendered != task.final_image:
                    issues.append(f"{tid}: true_order does not reconstruct final_image")
            if "answer_layers" in row:
                metrics = env.evaluate_layers(task, row["answer_layers"])
                if metrics["strict_success"] != 1.0:
                    issues.append(f"{tid}: answer_layers failed strict validation ({metrics})")
        except Exception as e:
            issues.append(f"{tid}: exception during validation: {e}")

    return {"count": len(rows), "issue_count": len(issues), "issues": issues}


if __name__ == "__main__":
    env = GeometricStackEnvironment(seed=7)
    task = env.generate_task(difficulty="medium", require_unique_order=True)
    print(env.format_task_for_llm(task))
    print("\nTrue order:", ",".join(task.true_order))
