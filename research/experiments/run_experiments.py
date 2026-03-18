"""
Run the 3-condition prompting experiment.

Features:
- JSON mode enforced with response_format={"type": "json_object"}
- Temperature fixed to 0 for deterministic outputs
- Retries include parse failures (up to 3 attempts)
- Structured failure codes:
    malformed_json | missing_key | too_few_blocks | api_error | refusal
- Reports both complete-case and ITT (failures=0) views
- Bootstrap 95% CI and paired Cohen's d
- Tracks valid_block_rate per condition
- Parallel execution via --parallel flag
- Incremental saving after each scenario completes

Usage:
    python -m research.experiments.run_experiments --test
    python -m research.experiments.run_experiments --full --parallel 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any

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
from research.metrics.cfs import cognitive_feasibility_score  # noqa: E402
from research.ontology.adhd_constraints import ADHDProfile, ScheduleBlock  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

MODEL = "openai/gpt-oss-120b"
SCENARIOS_PATH = PROJECT_ROOT / "research" / "scenarios" / "scenarios_50.json"
RESULTS_DIR = PROJECT_ROOT / "research" / "experiments" / "results"


class Condition(str, Enum):
    BASELINE = "baseline"
    ADHD_PROMPTED = "adhd_prompted"
    COGNISCHEDULE = "cognischedule"


class ErrorCode(str, Enum):
    MALFORMED_JSON = "malformed_json"
    MISSING_KEY = "missing_key"
    TOO_FEW_BLOCKS = "too_few_blocks"
    API_ERROR = "api_error"
    REFUSAL = "refusal"


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["scenarios"]


def extract_fixed_blocks(scenario: dict) -> list[dict]:
    return [b for b in scenario["schedule"] if b.get("is_fixed", False)]


def fixed_blocks_to_json(fixed_blocks: list[dict]) -> str:
    simplified = []
    for b in fixed_blocks:
        simplified.append(
            {
                "title": b["title"],
                "day": b["day"],
                "start_time": b["start_time"],
                "end_time": b["end_time"],
                "cognitive_load": b["cognitive_load"],
                "task_type": b["task_type"],
                "course": b.get("course"),
            }
        )
    return json.dumps(simplified, indent=2)


def build_messages(condition: Condition, scenario: dict) -> list[dict]:
    fixed = extract_fixed_blocks(scenario)
    fixed_json = fixed_blocks_to_json(fixed)
    week_context = scenario["week_context"]
    profile = scenario["profile"]

    if condition == Condition.BASELINE:
        return [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": baseline_user_prompt(week_context, fixed_json)},
        ]

    if condition == Condition.ADHD_PROMPTED:
        return [
            {"role": "system", "content": ADHD_PROMPTED_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": adhd_prompted_user_prompt(
                    week_context, fixed_json, profile["adhd_subtype"], profile["chronotype"]
                ),
            },
        ]

    if condition == Condition.COGNISCHEDULE:
        profile_json = json.dumps(profile, indent=2)
        return [
            {"role": "system", "content": COGNISCHEDULE_SYSTEM_PROMPT},
            {"role": "user", "content": cognischedule_user_prompt(week_context, fixed_json, profile_json)},
        ]

    raise ValueError(f"Unknown condition: {condition}")


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


def _looks_like_refusal(text: str) -> bool:
    low = text.lower()
    refusal_markers = [
        "i cannot",
        "i can't",
        "unable to",
        "cannot comply",
        "i'm sorry",
        "i am sorry",
        "as an ai",
        "i won't",
        "i will not",
    ]
    return any(marker in low for marker in refusal_markers)


def parse_schedule_json(raw: str | None) -> tuple[list[dict] | None, ErrorCode | None]:
    if raw is None or not raw.strip():
        return None, ErrorCode.REFUSAL

    text = _strip_markdown_fences(raw)

    # Try direct JSON first.
    data = _try_parse_json(text)

    # Fallback: extract largest object span.
    if data is None:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = _try_parse_json(text[start:end])

    if data is None:
        if _looks_like_refusal(text):
            return None, ErrorCode.REFUSAL
        return None, ErrorCode.MALFORMED_JSON

    if isinstance(data, dict):
        if "schedule" not in data:
            return None, ErrorCode.MISSING_KEY
        schedule = data["schedule"]
        if not isinstance(schedule, list):
            return None, ErrorCode.MALFORMED_JSON
        return schedule, None

    if isinstance(data, list):
        # Accept list form as legacy compatibility.
        return data, None

    return None, ErrorCode.MALFORMED_JSON


def validate_blocks(raw_blocks: list[dict]) -> tuple[list[ScheduleBlock], int, int]:
    valid: list[ScheduleBlock] = []
    invalid = 0
    raw_count = len(raw_blocks)
    for b in raw_blocks:
        try:
            valid.append(ScheduleBlock.model_validate(b))
        except Exception:
            invalid += 1
    return valid, raw_count, invalid


def call_model(client: Groq, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_completion_tokens=16384,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def run_trial(client: Groq, condition: Condition, scenario: dict, max_attempts: int = 3) -> dict:
    scenario_id = scenario["scenario_id"]
    profile = ADHDProfile.model_validate(scenario["profile"])
    messages = build_messages(condition, scenario)

    result = {
        "scenario_id": scenario_id,
        "condition": condition.value,
        "profile_id": profile.profile_id,
        "scenario_type": scenario["scenario_type"],
        "success": False,
        "attempts_used": 0,
        "error_code": None,
        "error_detail": None,
        "cfs_score": None,
        "violation_count": None,
        "total_penalty": None,
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
            raw_response = call_model(client, messages)
        except Exception as e:
            step["error_code"] = ErrorCode.API_ERROR.value
            step["error_detail"] = str(e)
            result["attempt_log"].append(step)
            result["error_code"] = ErrorCode.API_ERROR.value
            result["error_detail"] = str(e)
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

        cfs = cognitive_feasibility_score(blocks, profile)
        result["success"] = True
        result["attempts_used"] = attempt
        result["error_code"] = None
        result["error_detail"] = None
        result["cfs_score"] = round(float(cfs.score), 4)
        result["violation_count"] = int(cfs.violation_count)
        result["total_penalty"] = round(float(cfs.total_penalty), 4)
        result["block_count"] = len(blocks)
        result["attempt_log"].append(step)
        return result

    result["attempts_used"] = max_attempts
    return result


def _mean(values: list[float]) -> float | None:
    return round(float(statistics.mean(values)), 4) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return round(float(statistics.stdev(values)), 4)


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
    rng = random.Random(seed)
    n = len(a)
    diffs = [y - x for x, y in zip(a, b)]
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


def scenario_condition_map(results: list[dict]) -> dict[str, dict[str, dict]]:
    by_sid: dict[str, dict[str, dict]] = {}
    for r in results:
        by_sid.setdefault(r["scenario_id"], {})[r["condition"]] = r
    return by_sid


def pair_vectors(results: list[dict], cond_a: str, cond_b: str, itt: bool) -> tuple[list[float], list[float]]:
    by_sid = scenario_condition_map(results)
    a_vals: list[float] = []
    b_vals: list[float] = []
    for sid in sorted(by_sid):
        recs = by_sid[sid]
        if cond_a not in recs or cond_b not in recs:
            continue
        ra = recs[cond_a]
        rb = recs[cond_b]
        if itt:
            a_vals.append(float(ra["cfs_score"]) if ra["success"] else 0.0)
            b_vals.append(float(rb["cfs_score"]) if rb["success"] else 0.0)
        else:
            if ra["success"] and rb["success"]:
                a_vals.append(float(ra["cfs_score"]))
                b_vals.append(float(rb["cfs_score"]))
    return a_vals, b_vals


def view_summary(results: list[dict], itt: bool) -> dict:
    conditions = [c.value for c in Condition]
    summary: dict[str, dict] = {}
    for cond in conditions:
        vals: list[float] = []
        for r in results:
            if r["condition"] != cond:
                continue
            if r["success"] and r["cfs_score"] is not None:
                vals.append(float(r["cfs_score"]))
            elif itt:
                vals.append(0.0)
        summary[cond] = {
            "n": len(vals),
            "mean": _mean(vals),
            "std": _std(vals),
            "median": round(float(statistics.median(vals)), 4) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }
    return summary


def pairwise_stats(results: list[dict], itt: bool, n_boot: int) -> dict:
    comparisons = [
        ("baseline", "adhd_prompted"),
        ("baseline", "cognischedule"),
        ("adhd_prompted", "cognischedule"),
    ]
    out = {}
    for a, b in comparisons:
        av, bv = pair_vectors(results, a, b, itt=itt)
        key = f"{a}_vs_{b}"
        rep = wilcoxon_report(av, bv)
        rep["cohens_d_paired"] = paired_cohens_d(av, bv)
        lo, hi = bootstrap_ci_mean_diff(av, bv, n_boot=n_boot)
        rep["bootstrap_ci95_mean_diff"] = [lo, hi] if lo is not None else None
        out[key] = rep
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


def compute_report(results: list[dict], n_boot: int) -> dict:
    return {
        "config": {
            "model": MODEL,
            "temperature": 0,
            "json_mode": {"response_format": {"type": "json_object"}},
            "retry_on": ["api_error", "malformed_json", "missing_key", "too_few_blocks", "refusal"],
            "max_attempts_per_trial": 3,
            "bootstrap_samples": n_boot,
        },
        "complete_case": {
            "summary": view_summary(results, itt=False),
            "pairwise": pairwise_stats(results, itt=False, n_boot=n_boot),
        },
        "itt_failures_zero": {
            "summary": view_summary(results, itt=True),
            "pairwise": pairwise_stats(results, itt=True, n_boot=n_boot),
        },
        "reliability": reliability_stats(results),
    }


def print_report(report: dict) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY (COMPLETE-CASE VS ITT)")
    print("=" * 78)
    print("\nComplete-case means:")
    for cond, vals in report["complete_case"]["summary"].items():
        print(f"  {cond:<14} n={vals['n']:>2} mean={vals['mean']}")

    print("\nITT means (failures=0):")
    for cond, vals in report["itt_failures_zero"]["summary"].items():
        print(f"  {cond:<14} n={vals['n']:>2} mean={vals['mean']}")

    print("\nReliability:")
    print("  success_rates:", report["reliability"]["success_rates"])
    print("  valid_block_rate:", report["reliability"]["valid_block_rate"])
    print("  error_codes:", report["reliability"]["error_codes"])
    print("=" * 78)


def _run_scenario(
    client: Groq,
    scenario: dict,
    scenario_idx: int,
    total_scenarios: int,
    max_attempts: int,
) -> list[dict]:
    """Run all 3 conditions for a single scenario. Thread-safe."""
    conditions = [Condition.BASELINE, Condition.ADHD_PROMPTED, Condition.COGNISCHEDULE]
    sid = scenario["scenario_id"]
    results: list[dict] = []
    for cond in conditions:
        result = run_trial(client, cond, scenario, max_attempts=max_attempts)
        results.append(result)
        status = (
            f"OK cfs={result['cfs_score']:.3f} blocks={result['block_count']}"
            if result["success"]
            else f"FAIL code={result['error_code']}"
        )
        print(f"  [{scenario_idx}/{total_scenarios}] {sid} | {cond.value:<14} {status}", flush=True)
    return results


def _save_incremental(all_results: list[dict], test_mode: bool) -> None:
    """Save current results to disk (called after each scenario completes)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "test_" if test_mode else ""
    raw_path = RESULTS_DIR / f"{prefix}raw_results.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)


def run_experiment(
    test_mode: bool, max_attempts: int, n_boot: int, parallel: int = 1
) -> tuple[list[dict], dict]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key.strip())
    scenarios = load_scenarios()
    if test_mode:
        scenarios = scenarios[:1]

    total = len(scenarios)
    total_calls = total * 3
    mode = "TEST" if test_mode else "FULL"
    print(f"{mode} MODE: {total} scenarios x 3 conditions = {total_calls} API calls")
    print(f"Parallelism: {parallel} scenarios concurrently\n")

    all_results: list[dict] = []
    results_lock = threading.Lock()
    completed = 0

    if parallel <= 1:
        # Sequential (original behaviour)
        for i, scenario in enumerate(scenarios, start=1):
            scenario_results = _run_scenario(client, scenario, i, total, max_attempts)
            all_results.extend(scenario_results)
            _save_incremental(all_results, test_mode)
            completed += 1
            print(f"  -> Saved ({completed}/{total} scenarios done)\n", flush=True)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_run_scenario, client, sc, idx, total, max_attempts): idx
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
                    _save_incremental(all_results, test_mode)
                    print(f"  -> Saved ({completed}/{total} scenarios done)\n", flush=True)

    # Sort results by scenario_id for consistent ordering
    all_results.sort(key=lambda r: (r["scenario_id"], r["condition"]))

    report = compute_report(all_results, n_boot=n_boot)
    print_report(report)
    return all_results, report


def save_outputs(results: list[dict], report: dict, test_mode: bool) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "test_" if test_mode else ""
    raw_path = RESULTS_DIR / f"{prefix}raw_results.json"
    report_path = RESULTS_DIR / f"{prefix}table2_summary.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return raw_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-condition prompting experiment")
    parser.add_argument("--test", action="store_true", help="Run 1 scenario x 3 conditions")
    parser.add_argument("--full", action="store_true", help="Run 50 scenarios x 3 conditions (150 calls)")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts per trial")
    parser.add_argument("--bootstrap-samples", type=int, default=10000, help="Bootstrap samples for CI")
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of scenarios to run in parallel (e.g. 10 or 20)",
    )
    args = parser.parse_args()

    if not args.test and not args.full:
        print("Usage: python -m research.experiments.run_experiments --test|--full")
        sys.exit(1)

    test_mode = bool(args.test)
    results, report = run_experiment(
        test_mode=test_mode,
        max_attempts=args.max_attempts,
        n_boot=args.bootstrap_samples,
        parallel=args.parallel,
    )
    raw_path, report_path = save_outputs(results, report, test_mode=test_mode)
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {report_path}")


if __name__ == "__main__":
    main()

