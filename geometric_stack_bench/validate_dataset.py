"""
Validate geometric stack dataset rows using the same reconstruction/evaluation path.

Usage:
  python geometric_stack_bench/validate_dataset.py --dataset geometric_stack_bench/geometric_stack_dataset.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from geometric_stack_bench.geometric_stack_env import validate_dataset_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate geometric stack dataset answers and reconstructions.")
    parser.add_argument("--dataset", default="geometric_stack_bench/geometric_stack_dataset.json")
    args = parser.parse_args()

    report = validate_dataset_file(args.dataset)
    print(f"Rows checked: {report['count']}")
    print(f"Issues: {report['issue_count']}")
    if report["issues"]:
        print("\nFirst issues:")
        for msg in report["issues"][:20]:
            print(f"- {msg}")
        raise SystemExit(1)
    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
