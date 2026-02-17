"""
Evaluate Geometric Stack dataset using OpenRouter models.

Usage example (PowerShell):
  $env:OPENROUTER_API_KEY="..."
  python geometric_stack_bench\\openrouter_eval.py `
    --dataset geometric_stack_bench\\geometric_stack_dataset.json `
    --count 40 `
    --output geometric_stack_bench\\openrouter_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from geometric_stack_bench.geometric_stack_env import GeometricStackEnvironment, GeometricStackTask, Piece, ShapeType


DEFAULT_MODELS = [
    "openrouter/aurora-alpha",
    "openrouter/pony-alpha",
    "sourceful/riverflow-v2-pro",
    "stepfun/step-3.5-flash:free",
]


@dataclass
class ExampleResult:
    task_id: str
    predicted_text: str
    predicted_layers: List[Dict[str, Any]]
    predicted_order: List[str]
    metrics: Dict[str, float]
    error: Optional[str] = None


def row_to_task(row: Dict) -> GeometricStackTask:
    pieces = []
    for p in row["pieces"]:
        pieces.append(
            Piece(
                piece_id=p["piece_id"],
                shape=ShapeType(p["shape"]),
                color=p["color"],
                x=int(p["x"]),
                y=int(p["y"]),
                size=int(p["size"]),
            )
        )
    valid_orders = row.get("valid_orders")
    return GeometricStackTask(
        width=int(row["width"]),
        height=int(row["height"]),
        pieces=pieces,
        true_order=row.get("true_order", []),
        final_image=row["final_image"],
        difficulty=row["difficulty"],
        valid_order_count=int(row["valid_order_count"]),
        valid_orders=valid_orders,
    )


def call_openrouter(
    api_key: str,
    model: str,
    prompt: str,
    timeout_s: int = 60,
    max_retries: int = 3,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You may reason briefly, but you MUST end with exactly one answer box.\n"
                    "Required final format:\n"
                    '{"answer":[{"piece_id":"P1","shape":"square","x":2,"y":3,"z":0},'
                    '{"piece_id":"P2","shape":"triangle","x":5,"y":4,"z":1}]}\n'
                    "z=0 is bottom layer. No markdown code fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        req = urllib.request.Request(
            url=url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/linquert/kiln",
                "X-Title": "geometric-stack-eval",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                parsed = json.loads(body)
                return parsed["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {body}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(str(e))
    raise RuntimeError("OpenRouter request failed after retries")


def fetch_user_models(api_key: str, timeout_s: int = 60) -> List[str]:
    req = urllib.request.Request(
        url="https://openrouter.ai/api/v1/models/user",
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        return [m.get("id", "") for m in parsed.get("data", []) if m.get("id")]


def _extract_answer_box(text: str) -> Optional[str]:
    m = re.search(r"\{answer\s*:\s*(.*?)\}\s*$", text.strip(), flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"\{answer\s*:\s*(.*?)\}", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def parse_layers(text: str) -> List[Dict[str, Any]]:
    raw = text.strip()
    candidates: List[str] = [raw]

    boxed = _extract_answer_box(text)
    if boxed is not None:
        candidates.append(boxed)

    # JSON fenced block fallback.
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        candidates.append(m.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("answer"), list):
            out = [x for x in parsed["answer"] if isinstance(x, dict)]
            if out:
                return out
        if isinstance(parsed, list):
            out = [x for x in parsed if isinstance(x, dict)]
            if out:
                return out
    return []


def parse_order(text: str, expected_ids: List[str]) -> List[str]:
    block = _extract_answer_box(text)
    source = block if block is not None else text
    # Prefer strict token extraction like P12.
    ids = re.findall(r"\bP\d+\b", source.upper())
    if ids:
        seen = []
        for x in ids:
            if x not in seen:
                seen.append(x)
        return seen

    # Fallback: comma split.
    parts = [p.strip().upper() for p in source.split(",") if p.strip()]
    out = []
    for p in parts:
        if p not in out:
            out.append(p)
    return out


def summarize(results: List[ExampleResult]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {
            "count": 0,
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "avg_pixel_accuracy": 0.0,
            "avg_action_accuracy": 0.0,
            "error_rate": 0.0,
        }
    success = sum(r.metrics.get("success", 0.0) for r in results) / n
    avg_reward = sum(r.metrics.get("reward", 0.0) for r in results) / n
    avg_pixel = sum(r.metrics.get("pixel_accuracy", 0.0) for r in results) / n
    avg_action = sum(r.metrics.get("action_accuracy", 0.0) for r in results) / n
    error_rate = sum(1 for r in results if r.error) / n
    return {
        "count": n,
        "success_rate": success,
        "avg_reward": avg_reward,
        "avg_pixel_accuracy": avg_pixel,
        "avg_action_accuracy": avg_action,
        "error_rate": error_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate geometric stack dataset on OpenRouter models.")
    parser.add_argument("--dataset", default="geometric_stack_bench/geometric_stack_dataset.json")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", default="geometric_stack_bench/openrouter_eval_results.json")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--sleep-ms", type=int, default=200)
    parser.add_argument("--no-validate-models", action="store_true", help="Skip /models/user validation.")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-example console logging.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

    requested_models = args.models[:]
    if not args.no_validate_models:
        try:
            available = set(fetch_user_models(api_key))
            valid = [m for m in requested_models if m in available]
            invalid = [m for m in requested_models if m not in available]
            if invalid:
                print("Skipping unavailable models:")
                for m in invalid:
                    print(f"  - {m}")
            if not valid:
                raise RuntimeError("No requested models are available for this API key/account.")
            requested_models = valid
        except Exception as e:
            raise RuntimeError(f"Failed model validation via /models/user: {e}")

    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    selected = rows[args.offset : args.offset + args.count]
    env = GeometricStackEnvironment(seed=123)
    strict_tail = (
        "\n\nTASK:\n"
        "Infer placement order of pieces from BOTTOM -> TOP that reconstructs the final composite image.\n"
        "Use each piece exactly once.\n"
        "\nIMPORTANT:\n"
        "- This is an order reconstruction problem, not a shape description task.\n"
        "- A valid answer is any order that reproduces the final image exactly.\n"
        "\nVALID EXAMPLE OUTPUT:\n"
        '{"answer":[{"piece_id":"P3","shape":"circle","x":4,"y":6,"z":0},{"piece_id":"P1","shape":"square","x":1,"y":2,"z":1}]}\n'
        "\nINVALID OUTPUTS:\n"
        "- 'I think ...' before/after JSON  (extra text)\n"
        "- Missing z or wrong field names\n"
        "- Missing pieces or duplicate piece_id\n"
        "\nOUTPUT CONTRACT (REQUIRED):\n"
        "Return exactly one final JSON object:\n"
        '{"answer":[{"piece_id":"Pk","shape":"...","x":int,"y":int,"z":int}, ...]}\n'
        "No markdown fences.\n"
    )

    output = {
        "dataset": args.dataset,
        "offset": args.offset,
        "count": len(selected),
        "requested_models": args.models,
        "evaluated_models": requested_models,
        "models": {},
    }

    for model in requested_models:
        if not args.quiet:
            print(f"\n=== Model: {model} ===", flush=True)
        model_results: List[ExampleResult] = []
        for idx, row in enumerate(selected, 1):
            task = row_to_task(row)
            expected_ids = [p.piece_id for p in task.pieces]

            try:
                if not args.quiet:
                    print(f"[{idx}/{len(selected)}] task={row['id']} ...", flush=True)
                text = call_openrouter(api_key=api_key, model=model, prompt=row["prompt"] + strict_tail)
                pred_layers = parse_layers(text)
                if pred_layers:
                    pred = env._normalize_layers_to_order(pred_layers)
                    metrics = env.evaluate_layers(task, pred_layers)
                else:
                    pred = parse_order(text, expected_ids)
                    metrics = env.evaluate_order(task, pred)
                if not args.quiet:
                    print(f"  raw: {text[:200].replace(chr(10), ' | ')}", flush=True)
                    print(
                        f"  pred={pred} success={metrics.get('success')} reward={metrics.get('reward'):.3f} "
                        f"pixel={metrics.get('pixel_accuracy'):.3f} action={metrics.get('action_accuracy'):.3f}",
                        flush=True,
                    )
                model_results.append(
                    ExampleResult(
                        task_id=row["id"],
                        predicted_text=text,
                        predicted_layers=pred_layers if pred_layers else [],
                        predicted_order=pred,
                        metrics=metrics,
                        error=None,
                    )
                )
            except Exception as e:
                if not args.quiet:
                    print(f"  ERROR: {e}", flush=True)
                model_results.append(
                    ExampleResult(
                        task_id=row["id"],
                        predicted_text="",
                        predicted_layers=[],
                        predicted_order=[],
                        metrics={"success": 0.0, "reward": -1.0, "pixel_accuracy": 0.0, "action_accuracy": 0.0},
                        error=str(e),
                    )
                )
            time.sleep(max(args.sleep_ms, 0) / 1000.0)

        output["models"][model] = {
            "summary": summarize(model_results),
            "examples": [
                {
                    "task_id": r.task_id,
                    "predicted_text": r.predicted_text,
                    "predicted_layers": r.predicted_layers,
                    "predicted_order": r.predicted_order,
                    "metrics": r.metrics,
                    "error": r.error,
                }
                for r in model_results
            ],
        }

    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved evaluation to: {args.output}")
    for model, d in output["models"].items():
        s = d["summary"]
        print(
            f"{model} | n={s['count']} | success={s['success_rate']:.3f} | "
            f"reward={s['avg_reward']:.3f} | pixel={s['avg_pixel_accuracy']:.3f} | "
            f"action={s['avg_action_accuracy']:.3f} | error_rate={s['error_rate']:.3f}"
        )
        first_err = next((e.get("error") for e in d["examples"] if e.get("error")), None)
        if first_err:
            print(f"  first_error: {first_err}")


if __name__ == "__main__":
    main()
