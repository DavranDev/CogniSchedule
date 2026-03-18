"""
Schedule Adherence Probability (SAP) metric.

SAP is a secondary, non-CFS automated metric for estimating schedule
follow-through likelihood.

Unlike CFS, SAP does NOT use:
- trough-high-load rule checks
- buffer-threshold rule checks
- consecutive-high-load threshold checks
- monolithic-task rule checks

Instead, SAP estimates per-block completion probabilities from a different
feature set (duration fit, chronotype alignment, day congestion, course
fragmentation, decomposition support) and computes:
1) expected completion rate
2) probability of completing >=80% of actionable blocks
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from research.ontology.adhd_constraints import ADHDProfile, ScheduleBlock, SeverityLevel


ACTIONABLE_TYPES = {"study", "admin", "creative", "group_work", "lab", "exam", "lecture"}
NON_ACTIONABLE_TYPES = {"break", "exercise", "social"}

# SAP component labels (used for transparency and ablation analysis).
COMP_TIMING_ALIGNMENT = "timing_alignment"
COMP_SESSION_FIT = "session_fit"
COMP_DAY_ORGANIZATION = "day_organization"
COMP_PROFILE_FRICTION = "profile_friction"


@dataclass
class SAPResult:
    probability_complete_80: float
    expected_completion_rate: float
    actionable_block_count: int
    mean_block_probability: float


@dataclass
class SAPComponentSummary:
    timing_alignment: float
    session_fit: float
    day_organization: float
    profile_friction: float


def _time_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _duration_minutes(block: ScheduleBlock) -> int:
    start = _time_to_minutes(block.start_time)
    end = _time_to_minutes(block.end_time)
    return max(0, end - start)


def _start_hour(block: ScheduleBlock) -> int:
    return int(block.start_time.split(":")[0])


def _is_actionable(block: ScheduleBlock) -> bool:
    t = block.task_type.value
    if block.is_fixed:
        return False
    if t in NON_ACTIONABLE_TYPES:
        return False
    return t in ACTIONABLE_TYPES


def _chronotype_windows(chronotype: str) -> tuple[range, range]:
    # preferred, non-preferred
    if chronotype == "morning":
        return range(8, 14), range(19, 24)
    if chronotype == "evening":
        return range(16, 23), range(7, 11)
    return range(10, 18), range(7, 10)


def _severity_weight(level: SeverityLevel) -> float:
    if level == SeverityLevel.LOW:
        return 0.0
    if level == SeverityLevel.MODERATE:
        return 0.5
    return 1.0


def _base_by_load(cognitive_load: str) -> float:
    if cognitive_load == "low":
        return 0.82
    if cognitive_load == "medium":
        return 0.68
    return 0.54


def _daily_context(actionable: list[ScheduleBlock]) -> dict[str, dict]:
    by_day: dict[str, list[ScheduleBlock]] = {}
    for b in actionable:
        by_day.setdefault(b.day, []).append(b)

    out: dict[str, dict] = {}
    for day, blocks in by_day.items():
        blocks_sorted = sorted(blocks, key=lambda x: _time_to_minutes(x.start_time))
        distinct_courses = len({b.course for b in blocks_sorted if b.course})
        out[day] = {
            "count": len(blocks_sorted),
            "distinct_courses": distinct_courses,
        }
    return out


def _block_probability_with_components(
    block: ScheduleBlock,
    profile: ADHDProfile,
    day_ctx: dict,
    ablate_components: set[str] | None = None,
) -> tuple[float, dict[str, float]]:
    ablate = ablate_components or set()

    base = _base_by_load(block.cognitive_load.value)
    timing_alignment = 0.0
    session_fit = 0.0
    day_organization = 0.0
    profile_friction = 0.0

    # Duration fit around preferred study block duration.
    dur = _duration_minutes(block)
    pref = max(20, profile.preferred_study_block_minutes)
    dur_fit = math.exp(-abs(dur - pref) / pref)  # in (0,1]
    session_fit += 0.18 * (dur_fit - 0.5)

    # Chronotype alignment (coarse, independent from trough windows).
    preferred, non_preferred = _chronotype_windows(profile.chronotype.value)
    h = _start_hour(block)
    if h in preferred:
        timing_alignment += 0.08
    elif h in non_preferred:
        timing_alignment -= 0.08

    # Strong out-of-hours activation penalty.
    if h < 7 or h >= 23:
        timing_alignment -= 0.12

    # Day-level congestion and fragmentation penalties.
    day_count = day_ctx["count"]
    if day_count > 5:
        day_organization -= min(0.15, 0.03 * (day_count - 5))

    distinct_courses = day_ctx["distinct_courses"]
    if distinct_courses > 3:
        day_organization -= min(0.10, 0.02 * (distinct_courses - 3))

    # Decomposition support bonus.
    if block.is_decomposed:
        session_fit += 0.05

    # Trait-sensitive penalties that are not CFS rule copies.
    anx = _severity_weight(profile.anxiety_level)
    procrast = _severity_weight(profile.procrastination_tendency)
    if block.cognitive_load.value == "high":
        profile_friction -= 0.05 * anx
    if _start_hour(block) >= 19:
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

    return max(0.05, min(0.95, p)), {
        "timing_alignment": timing_alignment,
        "session_fit": session_fit,
        "day_organization": day_organization,
        "profile_friction": profile_friction,
    }


def schedule_adherence_probability(
    schedule: Iterable[ScheduleBlock],
    profile: ADHDProfile,
    n_simulations: int = 5000,
    seed: int = 42,
    ablate_components: Sequence[str] | None = None,
) -> SAPResult:
    """
    Estimate adherence from a generated schedule + profile.

    Returns:
    - probability_complete_80: P(complete >=80% actionable blocks)
    - expected_completion_rate: expected completed fraction across actionable blocks
    - actionable_block_count: number of blocks evaluated
    - mean_block_probability: mean Bernoulli probability per actionable block
    """
    blocks = list(schedule)
    actionable = [b for b in blocks if _is_actionable(b)]
    if not actionable:
        return SAPResult(
            probability_complete_80=1.0,
            expected_completion_rate=1.0,
            actionable_block_count=0,
            mean_block_probability=1.0,
        )

    ctx_by_day = _daily_context(actionable)
    ablate = set(ablate_components or [])
    probs = [_block_probability_with_components(b, profile, ctx_by_day[b.day], ablate)[0] for b in actionable]
    mean_prob = sum(probs) / len(probs)

    # Monte Carlo for P(complete >=80%).
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

    prob_complete_80 = success / n_simulations
    expected_completion_rate = mean_prob

    return SAPResult(
        probability_complete_80=round(prob_complete_80, 4),
        expected_completion_rate=round(expected_completion_rate, 4),
        actionable_block_count=len(probs),
        mean_block_probability=round(mean_prob, 4),
    )


def schedule_component_summary(
    schedule: Iterable[ScheduleBlock],
    profile: ADHDProfile,
) -> SAPComponentSummary:
    """
    Return mean additive contribution for each SAP component group.
    Positive values increase adherence probability; negative values decrease it.
    """
    blocks = list(schedule)
    actionable = [b for b in blocks if _is_actionable(b)]
    if not actionable:
        return SAPComponentSummary(
            timing_alignment=0.0,
            session_fit=0.0,
            day_organization=0.0,
            profile_friction=0.0,
        )

    ctx_by_day = _daily_context(actionable)
    comps = [
        _block_probability_with_components(b, profile, ctx_by_day[b.day])[1]
        for b in actionable
    ]
    n = len(comps)
    return SAPComponentSummary(
        timing_alignment=round(sum(c["timing_alignment"] for c in comps) / n, 4),
        session_fit=round(sum(c["session_fit"] for c in comps) / n, 4),
        day_organization=round(sum(c["day_organization"] for c in comps) / n, 4),
        profile_friction=round(sum(c["profile_friction"] for c in comps) / n, 4),
    )
