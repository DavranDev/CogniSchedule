"""
Compute Part 4 non-CFS evaluation (Table 3) from existing v2 experiment outputs.

Input:
  - research/experiments/results/raw_results.json
  - research/scenarios/scenarios_50.json

Output:
  - research/experiments/results/table3_sap_raw.json
  - research/experiments/results/table3_sap_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from research.metrics.sap import schedule_adherence_probability, schedule_component_summary
from research.ontology.adhd_constraints import ADHDProfile, ScheduleBlock

RAW_INPUT_DEFAULT = PROJECT_ROOT / "research" / "experiments" / "results" / "raw_results.json"
SCENARIOS_DEFAULT = PROJECT_ROOT / "research" / "scenarios" / "scenarios_50.json"
RAW_OUTPUT_DEFAULT = PROJECT_ROOT / "research" / "experiments" / "results" / "table3_sap_raw.json"
SUMMARY_OUTPUT_DEFAULT = PROJECT_ROOT / "research" / "experiments" / "results" / "table3_sap_summary.json"


def _strip_markdown_fences(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = t.split("\n", 1)[1] if "\n" in t else ""
    if t.endswith("```"):
        t = t[: t.rfind("```")]
    return t.strip()


def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_schedule_from_raw(raw: str | None) -> list[dict] | None:
    if raw is None or not raw.strip():
        return None
    text = _strip_markdown_fences(raw)
    data = _try_parse_json(text)
    if data is None:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = _try_parse_json(text[start:end])
    if data is None:
        return None
    if isinstance(data, dict):
        schedule = data.get("schedule")
        return schedule if isinstance(schedule, list) else None
    if isinstance(data, list):
        return data
    return None


def validate_blocks(raw_blocks: list[dict]) -> list[ScheduleBlock]:
    valid: list[ScheduleBlock] = []
    for b in raw_blocks:
        try:
            valid.append(ScheduleBlock.model_validate(b))
        except Exception:
            continue
    return valid


def paired_cohens_d(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    diffs = [y - x for x, y in zip(a, b)]
    sd = statistics.stdev(diffs)
    if sd == 0:
        return None
    return round(float(statistics.mean(diffs) / sd), 4)


def bootstrap_ci_mean_diff(
    a: list[float], b: list[float], n_boot: int = 10000, seed: int = 42
) -> tuple[float | None, float | None]:
    if len(a) != len(b) or not a:
        return None, None
    diffs = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return round(float(lo), 4), round(float(hi), 4)


def wilcoxon_report(a: list[float], b: list[float]) -> dict:
    if len(a) < 10:
        return {"n_paired": len(a), "note": "too_few_paired_samples"}
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return {"n_paired": len(a), "note": "scipy_not_available"}
    diffs = [y - x for x, y in zip(a, b)]
    if not any(d != 0 for d in diffs):
        return {"n_paired": len(a), "note": "no_variation"}
    stat, p = wilcoxon(a, b)
    return {
        "n_paired": len(a),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant_005": bool(p < 0.05),
        "mean_diff": round(float(statistics.mean(diffs)), 4),
    }


def _mean(vals: list[float]) -> float | None:
    return round(float(statistics.mean(vals)), 4) if vals else None


def _std(vals: list[float]) -> float | None:
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return round(float(statistics.stdev(vals)), 4)


def _metric_summary(rows: list[dict], metric_key: str, itt: bool) -> dict:
    conditions = ["baseline", "adhd_prompted", "cognischedule"]
    out: dict[str, dict] = {}
    for cond in conditions:
        vals: list[float] = []
        for r in rows:
            if r["condition"] != cond:
                continue
            ok = r["metric_success"]
            if ok:
                vals.append(float(r[metric_key]))
            elif itt:
                vals.append(0.0)
        out[cond] = {
            "n": len(vals),
            "mean": _mean(vals),
            "std": _std(vals),
            "median": round(float(statistics.median(vals)), 4) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }
    return out


def _pair_vectors(rows: list[dict], metric_key: str, cond_a: str, cond_b: str, itt: bool) -> tuple[list[float], list[float]]:
    by_sid: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_sid.setdefault(r["scenario_id"], {})[r["condition"]] = r

    a_vals: list[float] = []
    b_vals: list[float] = []
    for sid in sorted(by_sid.keys()):
        recs = by_sid[sid]
        if cond_a not in recs or cond_b not in recs:
            continue
        ra = recs[cond_a]
        rb = recs[cond_b]
        if itt:
            a_vals.append(float(ra[metric_key]) if ra["metric_success"] else 0.0)
            b_vals.append(float(rb[metric_key]) if rb["metric_success"] else 0.0)
        else:
            if ra["metric_success"] and rb["metric_success"]:
                a_vals.append(float(ra[metric_key]))
                b_vals.append(float(rb[metric_key]))
    return a_vals, b_vals


def _pairwise(rows: list[dict], metric_key: str, itt: bool, n_boot: int) -> dict:
    comparisons = [
        ("baseline", "adhd_prompted"),
        ("baseline", "cognischedule"),
        ("adhd_prompted", "cognischedule"),
    ]
    out = {}
    for a, b in comparisons:
        av, bv = _pair_vectors(rows, metric_key, a, b, itt=itt)
        rep = wilcoxon_report(av, bv)
        rep["cohens_d_paired"] = paired_cohens_d(av, bv)
        lo, hi = bootstrap_ci_mean_diff(av, bv, n_boot=n_boot)
        rep["bootstrap_ci95_mean_diff"] = [lo, hi] if lo is not None else None
        out[f"{a}_vs_{b}"] = rep
    return out


def load_profiles_map(scenarios_path: Path) -> dict[str, ADHDProfile]:
    obj = json.loads(scenarios_path.read_text(encoding="utf-8"))
    out: dict[str, ADHDProfile] = {}
    for s in obj["scenarios"]:
        out[s["scenario_id"]] = ADHDProfile.model_validate(s["profile"])
    return out


def compute_sap_rows(v2_rows: list[dict], profiles_by_sid: dict[str, ADHDProfile]) -> list[dict]:
    sap_rows: list[dict] = []
    for r in v2_rows:
        rec = {
            "scenario_id": r["scenario_id"],
            "condition": r["condition"],
            "metric_success": False,
            "metric_error": None,
            "sap_probability_complete_80": None,
            "sap_expected_completion_rate": None,
            "sap_actionable_block_count": 0,
            "sap_mean_block_probability": None,
            "sap_component_timing_alignment": None,
            "sap_component_session_fit": None,
            "sap_component_day_organization": None,
            "sap_component_profile_friction": None,
            "source_success": bool(r.get("success")),
            "source_error_code": r.get("error_code"),
        }

        if not r.get("success"):
            rec["metric_error"] = "source_failed"
            sap_rows.append(rec)
            continue

        raw_blocks = parse_schedule_from_raw(r.get("raw_response"))
        if raw_blocks is None:
            rec["metric_error"] = "postparse_failed"
            sap_rows.append(rec)
            continue

        blocks = validate_blocks(raw_blocks)
        if len(blocks) < 3:
            rec["metric_error"] = "too_few_valid_blocks"
            sap_rows.append(rec)
            continue

        profile = profiles_by_sid[r["scenario_id"]]
        sap = schedule_adherence_probability(blocks, profile)
        comps = schedule_component_summary(blocks, profile)
        rec["metric_success"] = True
        rec["sap_probability_complete_80"] = sap.probability_complete_80
        rec["sap_expected_completion_rate"] = sap.expected_completion_rate
        rec["sap_actionable_block_count"] = sap.actionable_block_count
        rec["sap_mean_block_probability"] = sap.mean_block_probability
        rec["sap_component_timing_alignment"] = comps.timing_alignment
        rec["sap_component_session_fit"] = comps.session_fit
        rec["sap_component_day_organization"] = comps.day_organization
        rec["sap_component_profile_friction"] = comps.profile_friction
        sap_rows.append(rec)

    return sap_rows


def compute_summary(rows: list[dict], n_boot: int) -> dict:
    metrics = {
        "sap_probability_complete_80": "sap_p80",
        "sap_expected_completion_rate": "sap_expected",
    }
    by_metric = {}
    for metric_key, label in metrics.items():
        by_metric[label] = {
            "complete_case": {
                "summary": _metric_summary(rows, metric_key, itt=False),
                "pairwise": _pairwise(rows, metric_key, itt=False, n_boot=n_boot),
            },
            "itt_failures_zero": {
                "summary": _metric_summary(rows, metric_key, itt=True),
                "pairwise": _pairwise(rows, metric_key, itt=True, n_boot=n_boot),
            },
        }

    conditions = ["baseline", "adhd_prompted", "cognischedule"]
    metric_success = {}
    metric_errors = {}
    for cond in conditions:
        subset = [r for r in rows if r["condition"] == cond]
        succ = sum(1 for r in subset if r["metric_success"])
        metric_success[cond] = f"{succ}/{len(subset)}"
        metric_errors[cond] = {}
        counts: dict[str, int] = {}
        for r in subset:
            if r["metric_success"]:
                continue
            err = r["metric_error"] or "unknown"
            counts[err] = counts.get(err, 0) + 1
        metric_errors[cond] = counts

    return {
        "config": {
            "input_file": str(RAW_INPUT_DEFAULT),
            "scenarios_file": str(SCENARIOS_DEFAULT),
            "bootstrap_samples": n_boot,
            "metric_note": (
                "SAP is a non-CFS metric based on adherence likelihood features, "
                "not CFS rule penalties."
            ),
        },
        "metric_compute_success": metric_success,
        "metric_compute_errors": metric_errors,
        "results": by_metric,
    }


def print_table3(summary: dict) -> None:
    sap = summary["results"]["sap_p80"]["complete_case"]["summary"]
    print("\n" + "=" * 78)
    print("TABLE 3 (NON-CFS): SAP P(complete >=80%)")
    print("=" * 78)
    for cond in ["baseline", "adhd_prompted", "cognischedule"]:
        s = sap[cond]
        print(f"{cond:<14} n={s['n']:>2} mean={s['mean']:.4f} std={s['std']:.4f}")
    print("metric_compute_success:", summary["metric_compute_success"])
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute non-CFS SAP evaluation (Table 3)")
    parser.add_argument("--input", type=Path, default=RAW_INPUT_DEFAULT, help="v2 raw results path")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_DEFAULT, help="scenarios_50 path")
    parser.add_argument("--out-raw", type=Path, default=RAW_OUTPUT_DEFAULT, help="SAP raw output path")
    parser.add_argument("--out-summary", type=Path, default=SUMMARY_OUTPUT_DEFAULT, help="SAP summary output path")
    parser.add_argument("--bootstrap-samples", type=int, default=10000, help="Bootstrap samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    v2_rows = json.loads(args.input.read_text(encoding="utf-8"))
    profiles = load_profiles_map(args.scenarios)

    sap_rows = compute_sap_rows(v2_rows, profiles)
    summary = compute_summary(sap_rows, n_boot=args.bootstrap_samples)

    args.out_raw.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_raw.write_text(json.dumps(sap_rows, indent=2), encoding="utf-8")
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_table3(summary)
    print(f"\nWrote SAP raw rows: {args.out_raw}")
    print(f"Wrote SAP summary:  {args.out_summary}")


if __name__ == "__main__":
    main()
