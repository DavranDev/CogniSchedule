"""
Cross-model robustness experiment (Table 4).

Runs the same 3-condition prompting experiment on additional model families
to demonstrate that CogniSchedule's CLT-grounded prompting effect is not
specific to a single model.

Features:
- Accepts --model flag to specify any Groq-hosted model
- Computes both CFS and SAP per trial (no separate post-hoc step)
- Generates per-model results and cross-model comparison table
- Reuses prompt templates, parsing, and statistical functions from main experiment

Usage:
    python -m research.experiments.robustness --model "llama-3.3-70b-versatile" --full --parallel 10
    python -m research.experiments.robustness --model "llama-3.1-8b-instant" --full --parallel 10
    python -m research.experiments.robustness --compare   # generate Table 4 from all results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.experiments.prompt_templates import (  # noqa: E402
    ADHD_PROMPTED_SYSTEM_PROMPT,
    BASELINE_SYSTEM_PROMPT,
    COGNISCHEDULE_SYSTEM_PROMPT,
    adhd_prompted_user_prompt,
    baseline_user_prompt,
    cognischedule_user_prompt,
)
from research.experiments.run_experiments import (  # noqa: E402
    Condition,
    ErrorCode,
    bootstrap_ci_mean_diff,
    build_messages,
    extract_fixed_blocks,
    load_scenarios,
    paired_cohens_d,
    parse_schedule_json,
    validate_blocks,
    wilcoxon_report,
)
from research.metrics.cfs import cognitive_feasibility_score  # noqa: E402
from research.metrics.sap import schedule_adherence_probability  # noqa: E402
from research.ontology.adhd_constraints import ADHDProfile  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

SCENARIOS_PATH = PROJECT_ROOT / "research" / "scenarios" / "scenarios_50.json"
RESULTS_DIR = PROJECT_ROOT / "research" / "experiments" / "results"

# Primary experiment model (for cross-model table)
PRIMARY_MODEL = "openai/gpt-oss-120b"


def _model_slug(model_id: str) -> str:
    """Convert model ID to filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def call_model(client: Groq, messages: list[dict], model: str, use_json_mode: bool = True) -> str:
    """Call a Groq-hosted model with optional JSON mode."""
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 16384,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    return content.strip() if content else ""


def run_trial(
    client: Groq,
    condition: Condition,
    scenario: dict,
    model: str,
    max_attempts: int = 3,
    use_json_mode: bool = True,
) -> dict:
    """Run a single trial: call model, parse, compute CFS + SAP."""
    scenario_id = scenario["scenario_id"]
    profile = ADHDProfile.model_validate(scenario["profile"])
    messages = build_messages(condition, scenario)

    result = {
        "scenario_id": scenario_id,
        "condition": condition.value,
        "model": model,
        "profile_id": profile.profile_id,
        "scenario_type": scenario["scenario_type"],
        "success": False,
        "attempts_used": 0,
        "error_code": None,
        "error_detail": None,
        # CFS
        "cfs_score": None,
        "violation_count": None,
        "total_penalty": None,
        # SAP
        "sap_probability_complete_80": None,
        "sap_expected_completion_rate": None,
        "sap_actionable_block_count": 0,
        "sap_mean_block_probability": None,
        # Metadata
        "block_count": 0,
        "raw_block_count": 0,
        "invalid_block_count": 0,
        "raw_response_length": 0,
        "raw_response": None,
        "attempt_log": [],
    }

    for attempt in range(1, max_attempts + 1):
        step = {"attempt": attempt, "error_code": None, "error_detail": None}
        try:
            raw_response = call_model(client, messages, model, use_json_mode=use_json_mode)
        except Exception as e:
            step["error_code"] = ErrorCode.API_ERROR.value
            step["error_detail"] = str(e)
            result["attempt_log"].append(step)
            result["error_code"] = ErrorCode.API_ERROR.value
            result["error_detail"] = str(e)
            # If JSON mode fails, retry without it
            if use_json_mode and "json" in str(e).lower():
                use_json_mode = False
            time.sleep(2)
            continue

        result["raw_response"] = raw_response
        result["raw_response_length"] = len(raw_response)

        raw_blocks, parse_err = parse_schedule_json(raw_response)
        if raw_blocks is None:
            step["error_code"] = parse_err.value if parse_err else ErrorCode.MALFORMED_JSON.value
            step["error_detail"] = "parse failed"
            result["attempt_log"].append(step)
            result["error_code"] = step["error_code"]
            result["error_detail"] = step["error_detail"]
            # If first attempt with JSON mode fails on parse, try without
            if use_json_mode and attempt == 1:
                use_json_mode = False
            time.sleep(1)
            continue

        blocks, raw_count, invalid_count = validate_blocks(raw_blocks)
        result["raw_block_count"] = raw_count
        result["invalid_block_count"] = invalid_count

        if len(blocks) < 3:
            step["error_code"] = ErrorCode.TOO_FEW_BLOCKS.value
            step["error_detail"] = f"valid_blocks={len(blocks)} raw_blocks={raw_count}"
            result["attempt_log"].append(step)
            result["error_code"] = step["error_code"]
            result["error_detail"] = step["error_detail"]
            time.sleep(1)
            continue

        # CFS
        cfs = cognitive_feasibility_score(blocks, profile)
        result["cfs_score"] = round(float(cfs.score), 4)
        result["violation_count"] = int(cfs.violation_count)
        result["total_penalty"] = round(float(cfs.total_penalty), 4)

        # SAP
        sap = schedule_adherence_probability(blocks, profile)
        result["sap_probability_complete_80"] = sap.probability_complete_80
        result["sap_expected_completion_rate"] = sap.expected_completion_rate
        result["sap_actionable_block_count"] = sap.actionable_block_count
        result["sap_mean_block_probability"] = sap.mean_block_probability

        result["success"] = True
        result["attempts_used"] = attempt
        result["error_code"] = None
        result["error_detail"] = None
        result["block_count"] = len(blocks)
        result["attempt_log"].append(step)
        return result

    result["attempts_used"] = max_attempts
    return result


def _run_scenario(
    client: Groq,
    scenario: dict,
    scenario_idx: int,
    total_scenarios: int,
    model: str,
    max_attempts: int,
) -> list[dict]:
    """Run all 3 conditions for a single scenario."""
    conditions = [Condition.BASELINE, Condition.ADHD_PROMPTED, Condition.COGNISCHEDULE]
    sid = scenario["scenario_id"]
    results: list[dict] = []
    for cond in conditions:
        result = run_trial(client, cond, scenario, model, max_attempts=max_attempts)
        results.append(result)
        if result["success"]:
            status = f"OK cfs={result['cfs_score']:.3f} sap={result['sap_probability_complete_80']:.3f} blocks={result['block_count']}"
        else:
            status = f"FAIL code={result['error_code']}"
        print(f"  [{scenario_idx}/{total_scenarios}] {sid} | {cond.value:<14} {status}", flush=True)
    return results


def _save_incremental(all_results: list[dict], slug: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"robustness_{slug}_raw.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)


# ============================================================================
# STATISTICS
# ============================================================================

def _mean(vals: list[float]) -> float | None:
    return round(float(statistics.mean(vals)), 4) if vals else None


def _std(vals: list[float]) -> float | None:
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return round(float(statistics.stdev(vals)), 4)


def condition_summary(results: list[dict], metric_key: str, itt: bool) -> dict:
    """Per-condition summary for a given metric."""
    conditions = [c.value for c in Condition]
    out: dict[str, dict] = {}
    for cond in conditions:
        vals: list[float] = []
        for r in results:
            if r["condition"] != cond:
                continue
            if r["success"] and r[metric_key] is not None:
                vals.append(float(r[metric_key]))
            elif itt:
                vals.append(0.0)
        out[cond] = {
            "n": len(vals),
            "mean": _mean(vals),
            "std": _std(vals),
            "median": round(float(statistics.median(vals)), 4) if vals else None,
            "min": round(min(vals), 4) if vals else None,
            "max": round(max(vals), 4) if vals else None,
        }
    return out


def pair_vectors(results: list[dict], metric_key: str, cond_a: str, cond_b: str, itt: bool) -> tuple[list[float], list[float]]:
    by_sid: dict[str, dict[str, dict]] = {}
    for r in results:
        by_sid.setdefault(r["scenario_id"], {})[r["condition"]] = r

    a_vals: list[float] = []
    b_vals: list[float] = []
    for sid in sorted(by_sid):
        recs = by_sid[sid]
        if cond_a not in recs or cond_b not in recs:
            continue
        ra = recs[cond_a]
        rb = recs[cond_b]
        if itt:
            a_vals.append(float(ra[metric_key]) if (ra["success"] and ra[metric_key] is not None) else 0.0)
            b_vals.append(float(rb[metric_key]) if (rb["success"] and rb[metric_key] is not None) else 0.0)
        else:
            if ra["success"] and rb["success"] and ra[metric_key] is not None and rb[metric_key] is not None:
                a_vals.append(float(ra[metric_key]))
                b_vals.append(float(rb[metric_key]))
    return a_vals, b_vals


def pairwise_stats(results: list[dict], metric_key: str, itt: bool, n_boot: int) -> dict:
    comparisons = [
        ("baseline", "adhd_prompted"),
        ("baseline", "cognischedule"),
        ("adhd_prompted", "cognischedule"),
    ]
    out = {}
    for a, b in comparisons:
        av, bv = pair_vectors(results, metric_key, a, b, itt=itt)
        rep = wilcoxon_report(av, bv)
        rep["cohens_d_paired"] = paired_cohens_d(av, bv)
        lo, hi = bootstrap_ci_mean_diff(av, bv, n_boot=n_boot)
        rep["bootstrap_ci95_mean_diff"] = [lo, hi] if lo is not None else None
        out[f"{a}_vs_{b}"] = rep
    return out


def reliability_stats(results: list[dict]) -> dict:
    conditions = [c.value for c in Condition]
    success_rates = {}
    errors = {}
    valid_block_rate = {}

    for cond in conditions:
        cond_rows = [r for r in results if r["condition"] == cond]
        succ = sum(1 for r in cond_rows if r["success"])
        success_rates[cond] = f"{succ}/{len(cond_rows)}"
        errors[cond] = dict(Counter(r["error_code"] for r in cond_rows if not r["success"]))

        raw_total = sum(int(r["raw_block_count"]) for r in cond_rows if r["raw_block_count"])
        valid_total = sum(int(r["block_count"]) for r in cond_rows if r["block_count"])
        rate = (valid_total / raw_total) if raw_total > 0 else None
        valid_block_rate[cond] = round(rate, 4) if rate is not None else None

    return {
        "success_rates": success_rates,
        "error_codes": errors,
        "valid_block_rate": valid_block_rate,
    }


def compute_model_report(results: list[dict], model: str, n_boot: int) -> dict:
    """Full report for a single model run."""
    report = {
        "config": {
            "model": model,
            "temperature": 0,
            "json_mode": True,
            "max_attempts_per_trial": 3,
            "bootstrap_samples": n_boot,
            "n_scenarios": len({r["scenario_id"] for r in results}),
            "n_trials": len(results),
        },
        "cfs": {
            "complete_case": {
                "summary": condition_summary(results, "cfs_score", itt=False),
                "pairwise": pairwise_stats(results, "cfs_score", itt=False, n_boot=n_boot),
            },
            "itt_failures_zero": {
                "summary": condition_summary(results, "cfs_score", itt=True),
                "pairwise": pairwise_stats(results, "cfs_score", itt=True, n_boot=n_boot),
            },
        },
        "sap_p80": {
            "complete_case": {
                "summary": condition_summary(results, "sap_probability_complete_80", itt=False),
                "pairwise": pairwise_stats(results, "sap_probability_complete_80", itt=False, n_boot=n_boot),
            },
            "itt_failures_zero": {
                "summary": condition_summary(results, "sap_probability_complete_80", itt=True),
                "pairwise": pairwise_stats(results, "sap_probability_complete_80", itt=True, n_boot=n_boot),
            },
        },
        "reliability": reliability_stats(results),
    }
    return report


# ============================================================================
# CROSS-MODEL COMPARISON (Table 4)
# ============================================================================

def load_primary_results() -> list[dict] | None:
    """Load the primary experiment's raw results for cross-model comparison."""
    path = RESULTS_DIR / "raw_results.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_model_row(results: list[dict], model: str, metric_key: str, n_boot: int) -> dict:
    """Extract a single row for the cross-model table."""
    summary = condition_summary(results, metric_key, itt=False)
    av, bv = pair_vectors(results, metric_key, "baseline", "cognischedule", itt=False)
    wil = wilcoxon_report(av, bv)
    d = paired_cohens_d(av, bv)
    lo, hi = bootstrap_ci_mean_diff(av, bv, n_boot=n_boot)

    # Determine effect direction
    b_mean = summary["baseline"]["mean"] or 0
    cs_mean = summary["cognischedule"]["mean"] or 0
    direction_holds = cs_mean > b_mean

    # Parse success rate
    total = len(results)
    success = sum(1 for r in results if r["success"])

    return {
        "model": model,
        "n_scenarios": len({r["scenario_id"] for r in results}),
        "parse_success": f"{success}/{total}",
        "parse_success_pct": round(100 * success / total, 1) if total > 0 else 0,
        "baseline_mean": summary["baseline"]["mean"],
        "baseline_std": summary["baseline"]["std"],
        "adhd_prompted_mean": summary["adhd_prompted"]["mean"],
        "adhd_prompted_std": summary["adhd_prompted"]["std"],
        "cognischedule_mean": summary["cognischedule"]["mean"],
        "cognischedule_std": summary["cognischedule"]["std"],
        "delta_c_minus_a": round(cs_mean - b_mean, 4) if (cs_mean and b_mean) else None,
        "cohens_d": d,
        "p_value": wil.get("p_value"),
        "bootstrap_ci95": [lo, hi] if lo is not None else None,
        "direction_holds": direction_holds,
        "ordering_b_lt_ap_lt_cs": (
            (summary["baseline"]["mean"] or 0)
            < (summary["adhd_prompted"]["mean"] or 0)
            < (summary["cognischedule"]["mean"] or 0)
        ),
    }


def generate_cross_model_table(n_boot: int = 10000) -> dict:
    """Generate Table 4 from all available robustness results + primary results."""
    models: list[dict] = []

    # Load primary model results
    primary = load_primary_results()
    if primary:
        # Primary results don't have SAP inline; compute from raw
        # For CFS, primary results use "cfs_score" key
        cfs_row = _extract_model_row(primary, PRIMARY_MODEL, "cfs_score", n_boot)
        models.append({"model": PRIMARY_MODEL, "cfs": cfs_row})

    # Load all robustness results
    for path in sorted(RESULTS_DIR.glob("robustness_*_raw.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            continue
        model_id = data[0].get("model", path.stem)
        cfs_row = _extract_model_row(data, model_id, "cfs_score", n_boot)
        sap_row = _extract_model_row(data, model_id, "sap_probability_complete_80", n_boot)
        models.append({"model": model_id, "cfs": cfs_row, "sap": sap_row})

    # Summary
    all_direction_holds = all(m["cfs"]["direction_holds"] for m in models)
    all_ordering_holds = all(m["cfs"]["ordering_b_lt_ap_lt_cs"] for m in models)

    table4 = {
        "table_name": "Table 4: Cross-Model Robustness",
        "description": "Effect direction consistency across model families and scales",
        "models": models,
        "summary": {
            "n_models": len(models),
            "all_direction_holds_cfs": all_direction_holds,
            "all_ordering_holds_cfs": all_ordering_holds,
        },
    }
    return table4


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_experiment(
    model: str,
    test_mode: bool,
    max_attempts: int,
    n_boot: int,
    parallel: int = 1,
) -> tuple[list[dict], dict]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key.strip())
    scenarios = load_scenarios()
    if test_mode:
        scenarios = scenarios[:2]

    slug = _model_slug(model)
    total = len(scenarios)
    total_calls = total * 3
    mode = "TEST" if test_mode else "FULL"
    print(f"\n{'='*78}")
    print(f"ROBUSTNESS EXPERIMENT: {model}")
    print(f"{mode} MODE: {total} scenarios x 3 conditions = {total_calls} API calls")
    print(f"Parallelism: {parallel} scenarios concurrently")
    print(f"{'='*78}\n")

    all_results: list[dict] = []
    results_lock = threading.Lock()
    completed = 0

    if parallel <= 1:
        for i, scenario in enumerate(scenarios, start=1):
            scenario_results = _run_scenario(client, scenario, i, total, model, max_attempts)
            all_results.extend(scenario_results)
            _save_incremental(all_results, slug)
            completed += 1
            print(f"  -> Saved ({completed}/{total} scenarios done)\n", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_run_scenario, client, sc, idx, total, model, max_attempts): idx
                for idx, sc in enumerate(scenarios, start=1)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    scenario_results = future.result()
                except Exception as e:
                    print(f"  [!] Scenario index {idx} raised exception: {e}", flush=True)
                    continue
                with results_lock:
                    all_results.extend(scenario_results)
                    completed += 1
                    _save_incremental(all_results, slug)
                    print(f"  -> Saved ({completed}/{total} scenarios done)\n", flush=True)

    # Sort for consistent ordering
    all_results.sort(key=lambda r: (r["scenario_id"], r["condition"]))

    report = compute_model_report(all_results, model, n_boot=n_boot)
    return all_results, report


def save_outputs(results: list[dict], report: dict, model: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _model_slug(model)
    raw_path = RESULTS_DIR / f"robustness_{slug}_raw.json"
    report_path = RESULTS_DIR / f"robustness_{slug}_summary.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return raw_path, report_path


def print_report(report: dict) -> None:
    model = report["config"]["model"]
    print(f"\n{'='*78}")
    print(f"RESULTS: {model}")
    print(f"{'='*78}")

    print("\nCFS (complete-case):")
    for cond, vals in report["cfs"]["complete_case"]["summary"].items():
        print(f"  {cond:<14} n={vals['n']:>2} mean={vals['mean']}")

    print("\nSAP P(>=80%) (complete-case):")
    for cond, vals in report["sap_p80"]["complete_case"]["summary"].items():
        print(f"  {cond:<14} n={vals['n']:>2} mean={vals['mean']}")

    print("\nReliability:")
    print("  success_rates:", report["reliability"]["success_rates"])
    print("  valid_block_rate:", report["reliability"]["valid_block_rate"])
    print("  error_codes:", report["reliability"]["error_codes"])

    # Effect direction check
    cfs_summary = report["cfs"]["complete_case"]["summary"]
    b = cfs_summary["baseline"]["mean"] or 0
    ap = cfs_summary["adhd_prompted"]["mean"] or 0
    cs = cfs_summary["cognischedule"]["mean"] or 0
    direction = "YES" if cs > b else "NO"
    ordering = "YES" if b < ap < cs else "NO"
    print(f"\nEffect direction (CS > B): {direction}")
    print(f"Full ordering (B < AP < CS): {ordering}")
    print(f"{'='*78}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model robustness experiment (Table 4)")
    parser.add_argument("--model", type=str, help="Groq model ID (e.g. llama-3.3-70b-versatile)")
    parser.add_argument("--test", action="store_true", help="Run 2 scenarios x 3 conditions (quick test)")
    parser.add_argument("--full", action="store_true", help="Run all 50 scenarios x 3 conditions")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts per trial")
    parser.add_argument("--bootstrap-samples", type=int, default=10000, help="Bootstrap CI samples")
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of scenarios to run in parallel",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Generate Table 4 cross-model comparison from existing results",
    )
    args = parser.parse_args()

    if args.compare:
        print("Generating cross-model comparison (Table 4)...")
        table4 = generate_cross_model_table(n_boot=args.bootstrap_samples)
        out_path = RESULTS_DIR / "table4_cross_model.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(table4, f, indent=2)

        print(f"\n{'='*78}")
        print("TABLE 4: CROSS-MODEL ROBUSTNESS")
        print(f"{'='*78}")
        for m in table4["models"]:
            cfs = m["cfs"]
            print(
                f"\n{cfs['model']:<35} "
                f"Parse={cfs['parse_success_pct']:>5.1f}%  "
                f"B={cfs['baseline_mean']}  AP={cfs['adhd_prompted_mean']}  "
                f"CS={cfs['cognischedule_mean']}  "
                f"d={cfs['cohens_d']}  "
                f"Direction={'YES' if cfs['direction_holds'] else 'NO'}"
            )
        print(f"\nAll directions hold (CFS): {table4['summary']['all_direction_holds_cfs']}")
        print(f"All orderings hold (CFS): {table4['summary']['all_ordering_holds_cfs']}")
        print(f"\nSaved to: {out_path}")
        return

    if not args.model:
        print("Error: --model is required (e.g. --model llama-3.3-70b-versatile)")
        sys.exit(1)

    if not args.test and not args.full:
        print("Usage: python -m research.experiments.robustness --model MODEL --test|--full")
        sys.exit(1)

    test_mode = bool(args.test)
    results, report = run_experiment(
        model=args.model,
        test_mode=test_mode,
        max_attempts=args.max_attempts,
        n_boot=args.bootstrap_samples,
        parallel=args.parallel,
    )
    raw_path, report_path = save_outputs(results, report, args.model)
    print_report(report)
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {report_path}")


if __name__ == "__main__":
    main()
