# Geometric Stack Bench

Code-verifiable RL environment for geometric occlusion reasoning.

This environment generates 2D composite images from layered geometric pieces and asks a model to recover the construction plan, not just the final visual similarity.

## What the environment evaluates

Given:
1. A fixed-size 2D canvas.
2. Pieces with known metadata: `piece_id`, `shape`, `color`, `x`, `y`, `size`.
3. A final overlapped composite image.

A model must return the layered construction as a list:
- `piece_id`
- `shape`
- `x`
- `y`
- `z` where `z=0` is bottom and larger `z` is nearer top.

This prevents “pixel-only hacks” where a model guesses an image-like answer without recovering the real construction ordering.

## Core design goals

1. Deterministic rendering and verification.
2. Standardized geometry and coordinate system.
3. Structured answer format with explicit z-order.
4. Order-first scoring, with visual match as fallback signal.
5. Dataset quality filters to avoid uninformative/ambiguous samples.

## Environment internals

### Shapes

Only three canonical shape classes are used:
1. `square`
2. `circle`
3. `triangle` (upright isosceles)

No custom polygons are used in dataset generation.

### Coordinate system

Two coordinate views are exposed:
1. Grid coordinates:
- Top-left cell is `(0,0)`.
- `x` increases to the right.
- `y` increases downward.
2. Axis coordinates:
- Origin at canvas center.
- `x` increases right.
- `y` increases upward.

### Anchor precision

Piece centers are constrained to half-grid precision only (e.g. `(0.5, 1.5)`, `(7.5, 4.5)`), never arbitrary floats.

How this is enforced:
1. Piece size is sampled from odd integers only.
2. Anchor is computed as `x + size/2`, `y + size/2`, which guarantees `.5`.

### Rendering semantics

1. Each piece generates a pixel mask.
2. Pieces are drawn in bottom-to-top order.
3. Later layers overwrite earlier layers.
4. Final image is exact deterministic output of this process.

This renderer is the single source of truth for both generation and evaluation.

## Task generation logic

`generate_task(...)` samples tasks by difficulty and filters them with quality constraints.

### Difficulty

Canvas is standardized to `16x16` for all difficulties.

Piece counts:
1. `easy`: 3
2. `medium`: 4
3. `hard`: 5
4. `expert`: 6

### Generation loop

For each candidate scene:
1. Sample pieces (shape, color, size, position).
2. Sample a random ground-truth order.
3. Render final image.
4. Compute overlap pixels.
5. Enforce quality constraints.
6. Count valid reconstructing orders by brute-force permutation.
7. Accept or reject based on uniqueness/ambiguity targets.

### Dataset quality constraints

The generator rejects poor samples using:
1. Minimum overlap threshold.
2. Minimum visible pixels per piece in final image.
3. Triangle signature visibility constraint.

## Triangle identifiability guarantee

To avoid triangle-vs-square ambiguity, each triangle must expose shape-specific evidence in the final image:
1. Apex visibility near tip.
2. At least one visible slanted boundary side.

If these cues are not visible due to occlusion, the sample is rejected.

This directly addresses the “triangle must show unique characteristic at least once” requirement.

## Evaluation logic

There are two evaluators:
1. `evaluate_order(...)`: order-only scoring from piece-id permutation.
2. `evaluate_layers(...)`: strict structured scoring from list of layer objects.

### Strict evaluator (`evaluate_layers`)

Primary checks:
1. Output length and `piece_id` set must match expected pieces.
2. Every row must include `piece_id`, `shape`, `x`, `y`, `z`.
3. Metadata correctness is scored (`shape`, `x`, `y` per piece).
4. Z-order is converted to order and scored against valid orders.

Then visual checks:
1. Render predicted order.
2. Measure exact success and pixel accuracy.

### Anti-gaming scoring policy

Reward prioritizes construction correctness:
1. If strict construction is correct (`valid order + metadata`), reward = `1.0`.
2. Otherwise reward is bounded `< 1` and weighted mostly by z-order and metadata, with only small pixel fallback.

This prevents LLMs from maximizing score via visual-only heuristics.

## Dataset schema (exported rows)

Each row includes:
1. Task metadata: `id`, `difficulty`, `width`, `height`.
2. Piece list with geometry, anchor, and axis coordinates.
3. `final_image`.
4. `valid_order_count`, `valid_orders`.
5. Ground truth:
- `true_order`
- `answer_layers` (structured canonical answer)
6. `prompt`.

Example answer layer format:

```json
[
  {"piece_id":"P1","shape":"square","x":4,"y":6,"z":0},
  {"piece_id":"P3","shape":"triangle","x":2,"y":5,"z":1},
  {"piece_id":"P2","shape":"circle","x":7,"y":4,"z":2}
]
```

## Validation and correctness checks

### Built-in dataset validation

`validate_dataset_file(...)` checks every row by replaying ground truth through the same reconstruction path:
1. Re-render `true_order` and verify exact image equality.
2. Run `evaluate_layers(answer_layers)` and require strict success.

CLI wrapper:

```bash
python geometric_stack_bench/validate_dataset.py --dataset geometric_stack_bench/geometric_stack_dataset.json
```

### Unit tests

Tests in `tests/test_geometric_stack_env.py` cover:
1. Unique-order correctness.
2. Ambiguous-order behavior.
3. Invalid-order penalties.
4. Structured layer scoring.
5. Metadata mismatch penalties.
6. Half-grid anchor precision.
7. Triangle signature visibility.
8. Export integrity + validator pass.

Run:

```bash
python -m unittest tests.test_geometric_stack_env -v
```

## End-to-end examples

### 1) Generate one task and inspect prompt

```python
from geometric_stack_bench.geometric_stack_env import GeometricStackEnvironment

env = GeometricStackEnvironment(seed=7)
task = env.generate_task(difficulty="medium", require_unique_order=True)
print(env.format_task_for_llm(task))
```

### 2) Score a perfect structured answer

```python
from geometric_stack_bench.geometric_stack_env import GeometricStackEnvironment

env = GeometricStackEnvironment(seed=7)
task = env.generate_task(difficulty="medium", require_unique_order=True)
layers = env.order_to_layers(task, task.true_order)
metrics = env.evaluate_layers(task, layers)
print(metrics)  # strict_success should be 1.0
```

### 3) Export a dataset with quality filters

```python
from geometric_stack_bench.geometric_stack_env import export_dataset

export_dataset(
    "geometric_stack_bench/geometric_stack_dataset.json",
    num_tasks=200,
    seed=42,
    require_unique_order=False,
    min_visible_pixels_per_piece=3,
    require_triangle_signature=True
)
```

### 4) Validate exported dataset

```bash
python geometric_stack_bench/validate_dataset.py --dataset geometric_stack_bench/geometric_stack_dataset.json
```

## OpenRouter evaluation notes

`openrouter_eval.py` now supports structured layer predictions:
1. It parses layer JSON outputs.
2. It evaluates with `evaluate_layers(...)` first.
3. It falls back to order-only parsing only when structured output is missing.

Use environment variable:

```bash
OPENROUTER_API_KEY=...
```

## Key files

1. `geometric_stack_bench/geometric_stack_env.py`
- Renderer
- Generator
- Structured and order evaluators
- Dataset export
- Dataset validator

2. `geometric_stack_bench/openrouter_eval.py`
- Model calling
- Output parsing
- Batch evaluation reporting

3. `geometric_stack_bench/validate_dataset.py`
- CLI validator for exported dataset files

4. `tests/test_geometric_stack_env.py`
- Core correctness and quality tests
