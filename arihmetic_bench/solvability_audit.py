"""
Brute-force solvability audit for CalcChain RLVR tasks.

Purpose:
- Sample tasks per structure/difficulty.
- Search for executable trajectories that satisfy verify semantics.
- Quantify solvable / unsat / unknown rates.

Unknown means search hit a resource cutoff (max nodes).
"""

import argparse
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Tuple

from calcchain_rlvr import CalcChainEnvironment, MathStructure, CalcChainTask


DIFFICULTIES = ["easy", "medium", "hard", "expert"]
COMMUTATIVE_OPS = {"+", "*", "gcd", "lcm"}


def _sorted_state(values):
    return tuple(sorted(values))


def _search_task_solvability(
    env: CalcChainEnvironment,
    task: CalcChainTask,
    max_nodes: int,
    max_abs_value: int,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "status": "solvable" | "unsat" | "unknown",
        "reason": str,
        "nodes": int,
      }
    """
    # state = (sorted_available_values, depth)
    q = deque()
    q.append((_sorted_state(task.numbers), 0))
    visited = set()
    nodes = 0

    while q:
        state, depth = q.popleft()
        key = (state, depth)
        if key in visited:
            continue
        visited.add(key)
        nodes += 1

        if nodes > max_nodes:
            return {"status": "unknown", "reason": "node_cutoff", "nodes": nodes}

        if depth >= task.max_steps:
            continue

        values = list(state)
        n = len(values)

        for op in task.operators:
            arity = env.operator_arity.get(op)
            if arity is None:
                continue

            local_next = set()

            if arity == 1:
                used_values = set()
                for i in range(n):
                    operand = values[i]
                    if operand in used_values:
                        continue
                    used_values.add(operand)

                    try:
                        result = env._apply_operation(op, [operand])
                    except Exception:
                        continue

                    if abs(result) > max_abs_value:
                        continue

                    new_values = values.copy()
                    new_values.pop(i)
                    new_values.append(result)
                    next_state = _sorted_state(new_values)

                    next_depth = depth + 1
                    if task.min_steps <= next_depth <= task.max_steps and result == task.target:
                        return {"status": "solvable", "reason": "found", "nodes": nodes}

                    local_next.add((next_state, next_depth))

            elif arity == 2:
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        if op in COMMUTATIVE_OPS and j < i:
                            continue

                        a, b = values[i], values[j]
                        try:
                            result = env._apply_operation(op, [a, b])
                        except Exception:
                            continue

                        if abs(result) > max_abs_value:
                            continue

                        new_values = values.copy()
                        for idx in sorted([i, j], reverse=True):
                            new_values.pop(idx)
                        new_values.append(result)
                        next_state = _sorted_state(new_values)

                        next_depth = depth + 1
                        if task.min_steps <= next_depth <= task.max_steps and result == task.target:
                            return {"status": "solvable", "reason": "found", "nodes": nodes}

                        local_next.add((next_state, next_depth))

            for item in local_next:
                if item not in visited:
                    q.append(item)

    return {"status": "unsat", "reason": "exhausted", "nodes": nodes}


def run_audit(
    samples_per_combo: int,
    seed: int,
    max_nodes: int,
    abs_factor: int,
    abs_floor: int,
    output_json: str = "",
) -> Dict[str, Any]:
    env = CalcChainEnvironment(seed=seed)

    report: Dict[str, Any] = {
        "meta": {
            "samples_per_combo": samples_per_combo,
            "seed": seed,
            "max_nodes": max_nodes,
            "abs_factor": abs_factor,
            "abs_floor": abs_floor,
            "structures": [s.value for s in MathStructure],
            "difficulties": DIFFICULTIES,
        },
        "by_combo": {},
        "summary": {},
    }

    summary_counter = Counter()
    total_nodes = 0
    total_tasks = 0

    for structure in MathStructure:
        for difficulty in DIFFICULTIES:
            combo_key = f"{structure.value}:{difficulty}"
            combo_counter = Counter()
            combo_nodes = 0
            unsat_examples = []
            unknown_examples = []

            for _ in range(samples_per_combo):
                task = env.generate_task(difficulty=difficulty, structure=structure)
                max_abs_value = max(abs_floor, abs(task.target) * abs_factor)
                result = _search_task_solvability(
                    env=env,
                    task=task,
                    max_nodes=max_nodes,
                    max_abs_value=max_abs_value,
                )

                status = result["status"]
                combo_counter[status] += 1
                combo_nodes += result["nodes"]
                summary_counter[status] += 1
                total_nodes += result["nodes"]
                total_tasks += 1

                if status == "unsat" and len(unsat_examples) < 3:
                    unsat_examples.append(
                        {
                            "target": task.target,
                            "numbers": task.numbers,
                            "operators": task.operators,
                            "min_steps": task.min_steps,
                            "max_steps": task.max_steps,
                            "structure_params": task.structure_params,
                        }
                    )
                if status == "unknown" and len(unknown_examples) < 3:
                    unknown_examples.append(
                        {
                            "reason": result["reason"],
                            "target": task.target,
                            "numbers": task.numbers,
                            "operators": task.operators,
                            "min_steps": task.min_steps,
                            "max_steps": task.max_steps,
                            "structure_params": task.structure_params,
                        }
                    )

            report["by_combo"][combo_key] = {
                "structure": structure.value,
                "difficulty": difficulty,
                "samples": samples_per_combo,
                "solvable": combo_counter["solvable"],
                "unsat": combo_counter["unsat"],
                "unknown": combo_counter["unknown"],
                "avg_nodes": combo_nodes / max(samples_per_combo, 1),
                "unsat_examples": unsat_examples,
                "unknown_examples": unknown_examples,
            }

    report["summary"] = {
        "total_tasks": total_tasks,
        "solvable": summary_counter["solvable"],
        "unsat": summary_counter["unsat"],
        "unknown": summary_counter["unknown"],
        "solvable_rate": summary_counter["solvable"] / max(total_tasks, 1),
        "unsat_rate": summary_counter["unsat"] / max(total_tasks, 1),
        "unknown_rate": summary_counter["unknown"] / max(total_tasks, 1),
        "avg_nodes_per_task": total_nodes / max(total_tasks, 1),
    }

    if output_json:
        out_path = Path(output_json)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def _print_report(report: Dict[str, Any]) -> None:
    print("=" * 100)
    print("CalcChain RLVR Brute-Force Solvability Audit")
    print("=" * 100)
    print("Config:", report["meta"])
    print()

    print(f"{'STRUCTURE:DIFF':52} {'SOLV':>5} {'UNSAT':>5} {'UNKN':>5} {'AVG_NODES':>10}")
    print("-" * 100)
    for combo_key, row in sorted(report["by_combo"].items()):
        print(
            f"{combo_key:52} "
            f"{row['solvable']:>5} "
            f"{row['unsat']:>5} "
            f"{row['unknown']:>5} "
            f"{row['avg_nodes']:>10.1f}"
        )

    print("-" * 100)
    s = report["summary"]
    print(
        "TOTAL:",
        f"tasks={s['total_tasks']}, solvable={s['solvable']} ({s['solvable_rate']:.2%}), "
        f"unsat={s['unsat']} ({s['unsat_rate']:.2%}), unknown={s['unknown']} ({s['unknown_rate']:.2%}), "
        f"avg_nodes={s['avg_nodes_per_task']:.1f}",
    )


def main():
    parser = argparse.ArgumentParser(description="Brute-force solvability audit for CalcChain RLVR.")
    parser.add_argument("--samples-per-combo", type=int, default=2, help="Tasks sampled per structure+difficulty.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for task sampling.")
    parser.add_argument("--max-nodes", type=int, default=40000, help="Per-task search node cutoff.")
    parser.add_argument(
        "--abs-factor",
        type=int,
        default=12,
        help="Intermediate absolute value cap multiplier: max(abs_floor, abs(target)*abs_factor).",
    )
    parser.add_argument("--abs-floor", type=int, default=3000, help="Minimum intermediate absolute value cap.")
    parser.add_argument("--output-json", type=str, default="calcchain_solvability_audit.json", help="Output path.")
    args = parser.parse_args()

    report = run_audit(
        samples_per_combo=args.samples_per_combo,
        seed=args.seed,
        max_nodes=args.max_nodes,
        abs_factor=args.abs_factor,
        abs_floor=args.abs_floor,
        output_json=args.output_json,
    )
    _print_report(report)
    print(f"\nSaved detailed report to: {args.output_json}")


if __name__ == "__main__":
    main()
