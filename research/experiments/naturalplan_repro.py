"""Build a non-matched NaturalPlan Table 1 with 95% Wilson confidence intervals.

This script is intentionally analysis-only. It does not call model APIs.
It consumes an existing summary CSV and produces a paper-ready CSV with CIs.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DEFAULT_INPUT = Path("test_figures/summary_table.csv")
DEFAULT_OUTPUT = Path("research/experiments/results/naturalplan_table1_nonmatched_with_ci.csv")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return Wilson score interval bounds in [0, 1]."""
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denom
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * total)) / total) / denom
    return center - margin, center + margin


def pct_to_count(pct: float, n: int) -> int:
    return int(round((pct / 100.0) * n))


def build_rows(
    input_rows: list[dict[str, str]],
    subset_n: int,
    comparison_type: str,
    note: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    overall_n = subset_n * 2

    for row in input_rows:
        model = row["model"]
        multi_day_pct = float(row["multi_day_2people"])
        multi_people_pct = float(row["multi_people_1day"])
        overall_pct = float(row["overall_accuracy_%"])

        multi_day_k = pct_to_count(multi_day_pct, subset_n)
        multi_people_k = pct_to_count(multi_people_pct, subset_n)
        overall_k = pct_to_count(overall_pct, overall_n)

        md_lo, md_hi = wilson_interval(multi_day_k, subset_n)
        mp_lo, mp_hi = wilson_interval(multi_people_k, subset_n)
        ov_lo, ov_hi = wilson_interval(overall_k, overall_n)

        rows.append(
            {
                "comparison_type": comparison_type,
                "model": model,
                "multi_day_accuracy_pct": f"{multi_day_pct:.1f}",
                "multi_day_n": str(subset_n),
                "multi_day_95ci_pct": f"[{md_lo * 100:.1f}, {md_hi * 100:.1f}]",
                "multi_people_accuracy_pct": f"{multi_people_pct:.1f}",
                "multi_people_n": str(subset_n),
                "multi_people_95ci_pct": f"[{mp_lo * 100:.1f}, {mp_hi * 100:.1f}]",
                "overall_accuracy_pct": f"{overall_pct:.1f}",
                "overall_n": str(overall_n),
                "overall_95ci_pct": f"[{ov_lo * 100:.1f}, {ov_hi * 100:.1f}]",
                "note": note,
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input summary CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument(
        "--subset-n",
        type=int,
        default=100,
        help="Examples per subset (default: 100; overall n is subset_n * 2)",
    )
    parser.add_argument(
        "--comparison-type",
        default="non_matched",
        help="Label for comparison type column (default: non_matched)",
    )
    parser.add_argument(
        "--note",
        default=(
            "Subset differs from NaturalPlan full 1,000-test setup; "
            "custom LLM-assisted extraction used"
        ),
        help="Note column text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        input_rows = list(reader)

    rows = build_rows(
        input_rows=input_rows,
        subset_n=args.subset_n,
        comparison_type=args.comparison_type,
        note=args.note,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
