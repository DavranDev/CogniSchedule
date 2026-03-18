"""
Supplementary analyses for paper-strengthening revisions.

Outputs:
  - research/experiments/results/table5_auxiliary_metrics.json
  - research/experiments/results/sap_component_ablation.json
  - research/experiments/results/qualitative_examples.json

This script intentionally uses only Python stdlib so it can run in lightweight
environments without the full research runtime dependencies.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "research" / "experiments" / "results"
RAW_RESULTS = RESULTS_DIR / "raw_results.json"
SAP_RAW_RESULTS = RESULTS_DIR / "table3_sap_raw.json"
SCENARIOS = PROJECT_ROOT / "research" / "scenarios" / "scenarios_50.json"

TABLE5_OUT = RESULTS_DIR / "table5_auxiliary_metrics.json"
SAP_ABLATION_OUT = RESULTS_DIR / "sap_component_ablation.json"
QUAL_OUT = RESULTS_DIR / "qualitative_examples.json"

ACTIONABLE_TYPES = {"study", "admin", "creative", "group_work", "lab", "exam", "lecture"}
NON_ACTIONABLE_TYPES = {"break", "exercise", "social"}

COMP_TIMING_ALIGNMENT = "timing_alignment"
COMP_SESSION_FIT = "session_fit"
COMP_DAY_ORGANIZATION = "day_organization"
COMP_PROFILE_FRICTION = "profile_friction"

DAY_ORDER = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _time_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _duration_minutes(block: dict[str, Any]) -> int:
    return max(0, _time_to_minutes(block["end_time"]) - _time_to_minutes(block["start_time"]))


def _strip_markdown_fences(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = t.split("\n", 1)[1] if "\n" in t else ""
    if t.endswith("```"):
        t = t[: t.rfind("```")]
    return t.strip()


def parse_schedule_from_raw(raw: str | None) -> list[dict[str, Any]] | None:
    if raw is None or not raw.strip():
        return None
    text = _strip_markdown_fences(raw)
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    if isinstance(data, dict):
        schedule = data.get("schedule")
        return schedule if isinstance(schedule, list) else None
    if isinstance(data, list):
        return data
    return None


def _is_actionable(block: dict[str, Any]) -> bool:
    task_type = block.get("task_type")
    if bool(block.get("is_fixed", False)):
        return False
    if task_type in NON_ACTIONABLE_TYPES:
        return False
    return task_type in ACTIONABLE_TYPES


def _sorted_actionable(schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actionable = [b for b in schedule if _is_actionable(b)]
    return sorted(
        actionable,
        key=lambda b: (DAY_ORDER.get(str(b.get("day", "monday")).lower(), 8), _time_to_minutes(b["start_time"])),
    )


def _tasks_are_dissimilar(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("task_type") != b.get("task_type"):
        return True
    course_a = a.get("course")
    course_b = b.get("course")
    return bool(course_a and course_b and course_a != course_b)


def _daily_actionable_loads(schedule: list[dict[str, Any]]) -> dict[str, int]:
    loads: dict[str, int] = {}
    for b in schedule:
        if not _is_actionable(b):
            continue
        day = str(b.get("day", "monday")).lower()
        loads[day] = loads.get(day, 0) + _duration_minutes(b)
    return loads


def _severity_weight(level: str) -> float:
    if level == "low":
        return 0.0
    if level == "moderate":
        return 0.5
    return 1.0


def _base_by_load(cognitive_load: str) -> float:
    if cognitive_load == "low":
        return 0.82
    if cognitive_load == "medium":
        return 0.68
    return 0.54


def _chronotype_windows(chronotype: str) -> tuple[set[int], set[int]]:
    if chronotype == "morning":
        return set(range(8, 14)), set(range(19, 24))
    if chronotype == "evening":
        return set(range(16, 23)), set(range(7, 11))
    return set(range(10, 18)), set(range(7, 10))


def _daily_context(actionable: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_day: dict[str, list[dict[str, Any]]] = {}
    for b in actionable:
        by_day.setdefault(str(b["day"]).lower(), []).append(b)

    out: dict[str, dict[str, int]] = {}
    for day, blocks in by_day.items():
        distinct_courses = len({b.get("course") for b in blocks if b.get("course")})
        out[day] = {"count": len(blocks), "distinct_courses": distinct_courses}
    return out


def _block_probability_with_components(
    block: dict[str, Any],
    profile: dict[str, Any],
    day_ctx: dict[str, int],
    ablate_components: set[str] | None = None,
) -> tuple[float, dict[str, float]]:
    ablate = ablate_components or set()
    base = _base_by_load(str(block.get("cognitive_load", "medium")))
    timing_alignment = 0.0
    session_fit = 0.0
    day_organization = 0.0
    profile_friction = 0.0

    dur = _duration_minutes(block)
    pref = max(20, int(profile.get("preferred_study_block_minutes", 45)))
    dur_fit = math.exp(-abs(dur - pref) / pref)
    session_fit += 0.18 * (dur_fit - 0.5)

    preferred, non_preferred = _chronotype_windows(str(profile.get("chronotype", "neutral")))
    hour = int(str(block["start_time"]).split(":")[0])
    if hour in preferred:
        timing_alignment += 0.08
    elif hour in non_preferred:
        timing_alignment -= 0.08
    if hour < 7 or hour >= 23:
        timing_alignment -= 0.12

    if day_ctx["count"] > 5:
        day_organization -= min(0.15, 0.03 * (day_ctx["count"] - 5))
    if day_ctx["distinct_courses"] > 3:
        day_organization -= min(0.10, 0.02 * (day_ctx["distinct_courses"] - 3))

    if bool(block.get("is_decomposed", False)):
        session_fit += 0.05

    anxiety = _severity_weight(str(profile.get("anxiety_level", "low")))
    procrast = _severity_weight(str(profile.get("procrastination_tendency", "moderate")))
    if str(block.get("cognitive_load")) == "high":
        profile_friction -= 0.05 * anxiety
    if hour >= 19:
        profile_friction -= 0.06 * procrast

    p = base
    if COMP_TIMING_ALIGNMENT not in ablate:
        p += timing_alignment
    if COMP_SESSION_FIT not in ablate:
        p += session_fit
    if COMP_DAY_ORGANIZATION not in ablate:
        p += day_organization
    if COMP_PROFILE_FRICTION not in ablate:
        p += profile_friction
    p = max(0.05, min(0.95, p))

    return p, {
        COMP_TIMING_ALIGNMENT: timing_alignment,
        COMP_SESSION_FIT: session_fit,
        COMP_DAY_ORGANIZATION: day_organization,
        COMP_PROFILE_FRICTION: profile_friction,
    }


@dataclass
class SAPRun:
    p80: float
    expected: float
    component_means: dict[str, float]


def schedule_adherence_probability(
    schedule: list[dict[str, Any]],
    profile: dict[str, Any],
    n_simulations: int = 5000,
    seed: int = 42,
    ablate_components: set[str] | None = None,
) -> SAPRun:
    actionable = [b for b in schedule if _is_actionable(b)]
    if not actionable:
        return SAPRun(
            p80=1.0,
            expected=1.0,
            component_means={
                COMP_TIMING_ALIGNMENT: 0.0,
                COMP_SESSION_FIT: 0.0,
                COMP_DAY_ORGANIZATION: 0.0,
                COMP_PROFILE_FRICTION: 0.0,
            },
        )

    ctx_by_day = _daily_context(actionable)
    probs: list[float] = []
    component_rows: list[dict[str, float]] = []
    for b in actionable:
        p, comps = _block_probability_with_components(
            b,
            profile,
            ctx_by_day[str(b["day"]).lower()],
            ablate_components=ablate_components,
        )
        probs.append(p)
        component_rows.append(comps)

    mean_prob = sum(probs) / len(probs)

    threshold = math.ceil(0.8 * len(probs))
    rng = random.Random(seed)
    success = 0
    for _ in range(n_simulations):
        completed = 0
        for p in probs:
            if rng.random() < p:
                completed += 1
        if completed >= threshold:
            success += 1

    component_means = {
        name: round(sum(row[name] for row in component_rows) / len(component_rows), 4)
        for name in [
            COMP_TIMING_ALIGNMENT,
            COMP_SESSION_FIT,
            COMP_DAY_ORGANIZATION,
            COMP_PROFILE_FRICTION,
        ]
    }
    return SAPRun(
        p80=round(success / n_simulations, 4),
        expected=round(mean_prob, 4),
        component_means=component_means,
    )


def _mean(vals: list[float]) -> float | None:
    return round(float(statistics.mean(vals)), 4) if vals else None


def _std(vals: list[float]) -> float | None:
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return round(float(statistics.stdev(vals)), 4)


def _median(vals: list[float]) -> float | None:
    return round(float(statistics.median(vals)), 4) if vals else None


def _bootstrap_ci_mean_diff(a: list[float], b: list[float], n_boot: int = 10000, seed: int = 42) -> list[float] | None:
    if len(a) != len(b) or not a:
        return None
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
    return [round(float(lo), 4), round(float(hi), 4)]


def _paired_values_by_scenario(rows: list[dict[str, Any]], metric_key: str, cond_a: str, cond_b: str) -> tuple[list[float], list[float]]:
    by_sid: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_sid.setdefault(r["scenario_id"], {})[r["condition"]] = r
    a_vals: list[float] = []
    b_vals: list[float] = []
    for sid in sorted(by_sid.keys()):
        recs = by_sid[sid]
        if cond_a in recs and cond_b in recs:
            a_vals.append(float(recs[cond_a][metric_key]))
            b_vals.append(float(recs[cond_b][metric_key]))
    return a_vals, b_vals


def _aux_metrics(schedule: list[dict[str, Any]]) -> dict[str, float]:
    actionable = _sorted_actionable(schedule)
    if not actionable:
        return {
            "transition_coherence": 0.0,
            "load_balance_score": 0.0,
            "course_continuity": 0.0,
        }

    # 1) Transition coherence (higher is better).
    if len(actionable) == 1:
        coherence = 1.0
        continuity = 0.0
    else:
        dissimilar = 0
        same_course = 0
        for i in range(len(actionable) - 1):
            a = actionable[i]
            b = actionable[i + 1]
            if _tasks_are_dissimilar(a, b):
                dissimilar += 1
            if a.get("course") and b.get("course") and a.get("course") == b.get("course"):
                same_course += 1
        transitions = len(actionable) - 1
        coherence = 1.0 - (dissimilar / transitions)
        continuity = same_course / transitions

    # 2) Load-balance score (higher is more balanced across days).
    loads = list(_daily_actionable_loads(schedule).values())
    if len(loads) <= 1:
        load_balance = 1.0
    else:
        std = statistics.stdev(loads)
        # Normalize with a conservative 120-minute scale.
        load_balance = 1.0 - min(1.0, std / 120.0)

    return {
        "transition_coherence": round(coherence, 4),
        "load_balance_score": round(load_balance, 4),
        "course_continuity": round(continuity, 4),
    }


def _is_break(block: dict[str, Any]) -> bool:
    return block.get("task_type") in {"break", "exercise", "social"}


def _is_in_trough(block: dict[str, Any], trough_hours: list[dict[str, int]]) -> bool:
    start_h = int(str(block["start_time"]).split(":")[0])
    end_h = int(str(block["end_time"]).split(":")[0])
    for tw in trough_hours:
        if start_h < int(tw["end_hour"]) and end_h > int(tw["start_hour"]):
            return True
    return False


def _cfs_rule_counts(schedule: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, int]:
    by_day: dict[str, list[dict[str, Any]]] = {}
    for b in schedule:
        by_day.setdefault(str(b.get("day", "monday")).lower(), []).append(b)
    for day in by_day:
        by_day[day].sort(key=lambda b: _time_to_minutes(b["start_time"]))

    r1 = r2 = r3 = r4 = 0
    min_buffer = int(profile.get("min_buffer_minutes", 10))
    trough_hours = list(profile.get("trough_hours", []))

    for blocks in by_day.values():
        # Rule 1: consecutive high-load >90 min.
        streak_minutes = 0
        streak_count = 0
        for b in blocks:
            if str(b.get("cognitive_load")) == "high" and not _is_break(b):
                streak_minutes += _duration_minutes(b)
                streak_count += 1
                if streak_minutes > 90 and streak_count >= 2:
                    r1 += 1
                    streak_minutes = 0
                    streak_count = 0
            else:
                streak_minutes = 0
                streak_count = 0

        # Rule 2: high-load in trough.
        for b in blocks:
            if str(b.get("cognitive_load")) == "high" and not _is_break(b) and _is_in_trough(b, trough_hours):
                r2 += 1

        # Rule 3: missing buffer between dissimilar tasks.
        for i in range(len(blocks) - 1):
            a = blocks[i]
            b = blocks[i + 1]
            if _is_break(a) or _is_break(b):
                continue
            if _tasks_are_dissimilar(a, b):
                gap = _time_to_minutes(b["start_time"]) - _time_to_minutes(a["end_time"])
                if gap < min_buffer:
                    r3 += 1

        # Rule 4: monolithic non-fixed task >120 min.
        for b in blocks:
            if (
                _duration_minutes(b) > 120
                and not bool(b.get("is_decomposed", False))
                and not _is_break(b)
                and b.get("task_type") not in {"lecture", "exam", "lab"}
            ):
                r4 += 1

    return {
        "consecutive_high": r1,
        "trough_high": r2,
        "missing_buffer": r3,
        "monolithic": r4,
    }


def run() -> None:
    raw_rows = json.loads(RAW_RESULTS.read_text(encoding="utf-8"))
    sap_rows = json.loads(SAP_RAW_RESULTS.read_text(encoding="utf-8"))
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"]
    profiles_by_sid = {s["scenario_id"]: s["profile"] for s in scenarios}

    analysis_rows: list[dict[str, Any]] = []
    for r in raw_rows:
        if not bool(r.get("success")):
            continue
        schedule = parse_schedule_from_raw(r.get("raw_response"))
        if not schedule:
            continue
        profile = profiles_by_sid[r["scenario_id"]]
        aux = _aux_metrics(schedule)
        sap_full = schedule_adherence_probability(schedule, profile, n_simulations=5000, seed=42)
        analysis_rows.append(
            {
                "scenario_id": r["scenario_id"],
                "condition": r["condition"],
                "scenario_type": r["scenario_type"],
                "profile_id": r["profile_id"],
                "transition_coherence": aux["transition_coherence"],
                "load_balance_score": aux["load_balance_score"],
                "course_continuity": aux["course_continuity"],
                "sap_p80_recomputed": sap_full.p80,
                "sap_expected_recomputed": sap_full.expected,
                "sap_components": sap_full.component_means,
                "cfs_score": r["cfs_score"],
            }
        )

    conditions = ["baseline", "adhd_prompted", "cognischedule"]
    aux_metrics = ["transition_coherence", "load_balance_score", "course_continuity"]
    table5: dict[str, Any] = {"metrics": {}, "rows": len(analysis_rows)}

    for metric in aux_metrics:
        summary = {}
        for cond in conditions:
            vals = [float(r[metric]) for r in analysis_rows if r["condition"] == cond]
            summary[cond] = {
                "n": len(vals),
                "mean": _mean(vals),
                "std": _std(vals),
                "median": _median(vals),
            }
        pairwise = {}
        for a, b in [("baseline", "adhd_prompted"), ("baseline", "cognischedule"), ("adhd_prompted", "cognischedule")]:
            av, bv = _paired_values_by_scenario(analysis_rows, metric, a, b)
            diffs = [y - x for x, y in zip(av, bv)]
            pairwise[f"{a}_vs_{b}"] = {
                "n_paired": len(av),
                "mean_diff": round(float(statistics.mean(diffs)), 4) if diffs else None,
                "bootstrap_ci95_mean_diff": _bootstrap_ci_mean_diff(av, bv, n_boot=10000, seed=42),
            }
        table5["metrics"][metric] = {"summary": summary, "pairwise": pairwise}

    TABLE5_OUT.write_text(json.dumps(table5, indent=2), encoding="utf-8")

    # SAP component ablation
    by_sid_cond: dict[str, dict[str, dict[str, Any]]] = {}
    for r in analysis_rows:
        by_sid_cond.setdefault(r["scenario_id"], {})[r["condition"]] = r

    ablation_components = [
        COMP_TIMING_ALIGNMENT,
        COMP_SESSION_FIT,
        COMP_DAY_ORGANIZATION,
        COMP_PROFILE_FRICTION,
    ]
    ablation_rows: list[dict[str, Any]] = []
    for r in raw_rows:
        if r.get("condition") != "cognischedule" or not bool(r.get("success")):
            continue
        schedule = parse_schedule_from_raw(r.get("raw_response"))
        if not schedule:
            continue
        profile = profiles_by_sid[r["scenario_id"]]
        full = schedule_adherence_probability(schedule, profile, n_simulations=5000, seed=42)
        entry = {
            "scenario_id": r["scenario_id"],
            "full_p80": full.p80,
            "full_expected": full.expected,
            "full_components": full.component_means,
            "ablations": {},
        }
        for comp in ablation_components:
            abl = schedule_adherence_probability(schedule, profile, n_simulations=5000, seed=42, ablate_components={comp})
            entry["ablations"][comp] = {
                "p80": abl.p80,
                "expected": abl.expected,
                "delta_p80_full_minus_ablated": round(full.p80 - abl.p80, 4),
                "delta_expected_full_minus_ablated": round(full.expected - abl.expected, 4),
            }
        ablation_rows.append(entry)

    summary = {"n": len(ablation_rows), "components": {}}
    for comp in ablation_components:
        deltas_p80 = [float(r["ablations"][comp]["delta_p80_full_minus_ablated"]) for r in ablation_rows]
        deltas_expected = [float(r["ablations"][comp]["delta_expected_full_minus_ablated"]) for r in ablation_rows]
        summary["components"][comp] = {
            "mean_delta_p80_full_minus_ablated": _mean(deltas_p80),
            "mean_delta_expected_full_minus_ablated": _mean(deltas_expected),
            "bootstrap_ci95_delta_p80": _bootstrap_ci_mean_diff(
                [float(r["ablations"][comp]["p80"]) for r in ablation_rows],
                [float(r["full_p80"]) for r in ablation_rows],
                n_boot=10000,
                seed=42,
            ),
        }

    SAP_ABLATION_OUT.write_text(
        json.dumps({"summary": summary, "rows": ablation_rows}, indent=2),
        encoding="utf-8",
    )

    # Qualitative examples: one strong improvement + one difficult failure.
    sap_by_sid_cond: dict[str, dict[str, dict[str, Any]]] = {}
    for r in sap_rows:
        sap_by_sid_cond.setdefault(r["scenario_id"], {})[r["condition"]] = r

    scenario_deltas = []
    for sid, recs in by_sid_cond.items():
        if not all(c in recs for c in conditions):
            continue
        sap_recs = sap_by_sid_cond.get(sid, {})
        if not all(c in sap_recs for c in ["baseline", "cognischedule"]):
            continue
        cfs_delta = float(recs["cognischedule"]["cfs_score"]) - float(recs["baseline"]["cfs_score"])
        sap_delta = float(sap_recs["cognischedule"]["sap_probability_complete_80"]) - float(
            sap_recs["baseline"]["sap_probability_complete_80"]
        )
        scenario_deltas.append((sid, cfs_delta, sap_delta))

    best = max(scenario_deltas, key=lambda x: (x[1], x[2]))
    hardest = min(scenario_deltas, key=lambda x: x[2])

    def build_example(sid: str, label: str) -> dict[str, Any]:
        profile = profiles_by_sid[sid]
        out = {"label": label, "scenario_id": sid, "conditions": {}}
        for cond in ["baseline", "cognischedule"]:
            raw_rec = next(r for r in raw_rows if r["scenario_id"] == sid and r["condition"] == cond)
            sap_rec = next(r for r in sap_rows if r["scenario_id"] == sid and r["condition"] == cond)
            schedule = parse_schedule_from_raw(raw_rec.get("raw_response")) or []
            out["conditions"][cond] = {
                "cfs_score": raw_rec["cfs_score"],
                "sap_p80": sap_rec["sap_probability_complete_80"],
                "rule_counts": _cfs_rule_counts(schedule, profile),
                "actionable_blocks": sum(1 for b in schedule if _is_actionable(b)),
                "high_load_actionable_blocks": sum(
                    1 for b in schedule if _is_actionable(b) and str(b.get("cognitive_load")) == "high"
                ),
            }
        return out

    QUAL_OUT.write_text(
        json.dumps(
            {
                "best_improvement": build_example(best[0], "largest_cfs_gain"),
                "hard_failure_case": build_example(hardest[0], "largest_negative_sap_delta"),
                "selection_basis": {
                    "best": {"cfs_delta": round(best[1], 4), "sap_delta": round(best[2], 4)},
                    "hardest": {"cfs_delta": round(hardest[1], 4), "sap_delta": round(hardest[2], 4)},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote auxiliary metrics: {TABLE5_OUT}")
    print(f"Wrote SAP ablation:      {SAP_ABLATION_OUT}")
    print(f"Wrote qualitative cases: {QUAL_OUT}")


if __name__ == "__main__":
    run()
