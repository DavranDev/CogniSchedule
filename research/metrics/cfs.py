"""
Cognitive Feasibility Score (CFS) — Automated ADHD schedule evaluation metric.

CFS(schedule, adhd_profile) → float [0, 1]

Rule-based penalty system (no annotation or clinical validation required):
  1. Consecutive high-CLC blocks > 90 min without break:  −0.15 per violation
  2. High-CLC task during circadian trough:                −0.10 per violation
  3. Missing transition buffer between dissimilar tasks:   −0.05 per violation
  4. Monolithic task > 2 hours without decomposition:      −0.10 per violation

Framing: "We propose CFS as an automated evaluation proxy pending clinical validation."

Usage:
    from research.metrics.cfs import cognitive_feasibility_score
    score, violations = cognitive_feasibility_score(schedule, profile)
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.ontology.adhd_constraints import (
    ADHDProfile,
    CognitiveLoadCategory,
    ScheduleBlock,
    TimeWindow,
)


# ============================================================================
# PENALTY WEIGHTS
# ============================================================================

PENALTY_CONSECUTIVE_HIGH_LOAD = 0.15
PENALTY_TROUGH_HIGH_LOAD = 0.10
PENALTY_MISSING_BUFFER = 0.05
PENALTY_MONOLITHIC_TASK = 0.10


@dataclass
class Violation:
    """A single CFS violation with context for debugging/reporting."""
    rule: str
    penalty: float
    description: str
    day: str = ""
    block_title: str = ""


@dataclass
class CFSResult:
    """Full CFS evaluation result."""
    score: float
    violations: list[Violation] = field(default_factory=list)
    total_penalty: float = 0.0
    block_count: int = 0
    days_evaluated: int = 0

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def summary(self) -> str:
        lines = [
            f"CFS Score: {self.score:.3f}",
            f"Violations: {self.violation_count} (total penalty: {self.total_penalty:.3f})",
            f"Blocks evaluated: {self.block_count} across {self.days_evaluated} days",
        ]
        if self.violations:
            lines.append("Details:")
            for v in self.violations:
                lines.append(f"  [{v.rule}] -{v.penalty:.2f}: {v.description}")
        return "\n".join(lines)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _parse_time_minutes(time_str: str) -> int:
    """Convert HH:MM to minutes since midnight."""
    h, m = time_str.split(":")
    return int(h) * 60 + int(m)


def _block_duration_minutes(block: ScheduleBlock) -> int:
    """Duration of a schedule block in minutes."""
    start = _parse_time_minutes(block.start_time)
    end = _parse_time_minutes(block.end_time)
    return max(0, end - start)


def _block_start_hour(block: ScheduleBlock) -> int:
    """Start hour of a block."""
    return int(block.start_time.split(":")[0])


def _is_in_trough(block: ScheduleBlock, trough_hours: list[TimeWindow]) -> bool:
    """Check if any part of the block falls within a trough window."""
    start_h = _block_start_hour(block)
    end_h = int(block.end_time.split(":")[0])
    for tw in trough_hours:
        if start_h < tw.end_hour and end_h > tw.start_hour:
            return True
    return False


def _is_break(block: ScheduleBlock) -> bool:
    """Check if a block is a break/rest/non-cognitive activity."""
    return block.task_type.value in ("break", "exercise", "social")


def _tasks_are_dissimilar(a: ScheduleBlock, b: ScheduleBlock) -> bool:
    """Two tasks are dissimilar if they differ in both type and course."""
    if a.task_type != b.task_type:
        return True
    if a.course and b.course and a.course != b.course:
        return True
    return False


def _gap_minutes(a: ScheduleBlock, b: ScheduleBlock) -> int:
    """Gap in minutes between end of a and start of b."""
    end_a = _parse_time_minutes(a.end_time)
    start_b = _parse_time_minutes(b.start_time)
    return start_b - end_a


# ============================================================================
# PENALTY CHECKERS
# ============================================================================

def _check_consecutive_high_load(
    day_blocks: list[ScheduleBlock],
    day: str,
) -> list[Violation]:
    """
    Rule 1: Consecutive high-CLC blocks totaling > 90 min without a break.
    """
    violations = []
    consecutive_high_minutes = 0
    streak_titles = []

    for block in day_blocks:
        if block.cognitive_load == CognitiveLoadCategory.HIGH and not _is_break(block):
            consecutive_high_minutes += _block_duration_minutes(block)
            streak_titles.append(block.title)

            if consecutive_high_minutes > 90 and len(streak_titles) >= 2:
                violations.append(Violation(
                    rule="consecutive_high_load",
                    penalty=PENALTY_CONSECUTIVE_HIGH_LOAD,
                    description=(
                        f"Consecutive high-load blocks exceed 90 min "
                        f"({consecutive_high_minutes} min): {', '.join(streak_titles)}"
                    ),
                    day=day,
                    block_title=streak_titles[-1],
                ))
                # Reset to avoid double-counting the same streak
                consecutive_high_minutes = 0
                streak_titles = []
        else:
            consecutive_high_minutes = 0
            streak_titles = []

    return violations


def _check_trough_high_load(
    day_blocks: list[ScheduleBlock],
    day: str,
    trough_hours: list[TimeWindow],
) -> list[Violation]:
    """
    Rule 2: High-CLC task scheduled during circadian trough.
    """
    violations = []
    for block in day_blocks:
        if (
            block.cognitive_load == CognitiveLoadCategory.HIGH
            and not _is_break(block)
            and _is_in_trough(block, trough_hours)
        ):
            violations.append(Violation(
                rule="trough_high_load",
                penalty=PENALTY_TROUGH_HIGH_LOAD,
                description=(
                    f"High-load task '{block.title}' at {block.start_time}-{block.end_time} "
                    f"falls within circadian trough"
                ),
                day=day,
                block_title=block.title,
            ))
    return violations


def _check_missing_buffers(
    day_blocks: list[ScheduleBlock],
    day: str,
    min_buffer: int,
) -> list[Violation]:
    """
    Rule 3: Missing transition buffer (< min_buffer minutes) between dissimilar tasks.
    """
    violations = []
    for i in range(len(day_blocks) - 1):
        a, b = day_blocks[i], day_blocks[i + 1]

        if _is_break(a) or _is_break(b):
            continue

        if _tasks_are_dissimilar(a, b):
            gap = _gap_minutes(a, b)
            if gap < min_buffer:
                violations.append(Violation(
                    rule="missing_buffer",
                    penalty=PENALTY_MISSING_BUFFER,
                    description=(
                        f"Only {gap} min buffer between dissimilar tasks "
                        f"'{a.title}' and '{b.title}' (need {min_buffer} min)"
                    ),
                    day=day,
                    block_title=b.title,
                ))
    return violations


def _check_monolithic_tasks(
    day_blocks: list[ScheduleBlock],
    day: str,
) -> list[Violation]:
    """
    Rule 4: Single task > 2 hours (120 min) without decomposition.
    """
    violations = []
    for block in day_blocks:
        duration = _block_duration_minutes(block)
        if (
            duration > 120
            and not block.is_decomposed
            and not _is_break(block)
            and block.task_type.value not in ("lecture", "exam", "lab")
        ):
            violations.append(Violation(
                rule="monolithic_task",
                penalty=PENALTY_MONOLITHIC_TASK,
                description=(
                    f"Task '{block.title}' is {duration} min without decomposition"
                ),
                day=day,
                block_title=block.title,
            ))
    return violations


# ============================================================================
# MAIN SCORING FUNCTION
# ============================================================================

def cognitive_feasibility_score(
    schedule: list[ScheduleBlock],
    profile: ADHDProfile,
) -> CFSResult:
    """
    Compute Cognitive Feasibility Score for a schedule given an ADHD profile.

    Returns CFSResult with score in [0, 1] (1.0 = no violations, 0.0 = floor).
    """
    # Group blocks by day and sort by start time
    days: dict[str, list[ScheduleBlock]] = {}
    for block in schedule:
        days.setdefault(block.day, []).append(block)

    for day in days:
        days[day].sort(key=lambda b: _parse_time_minutes(b.start_time))

    all_violations: list[Violation] = []

    for day, day_blocks in days.items():
        all_violations.extend(_check_consecutive_high_load(day_blocks, day))
        all_violations.extend(_check_trough_high_load(day_blocks, day, profile.trough_hours))
        all_violations.extend(_check_missing_buffers(day_blocks, day, profile.min_buffer_minutes))
        all_violations.extend(_check_monolithic_tasks(day_blocks, day))

    total_penalty = sum(v.penalty for v in all_violations)
    score = max(0.0, 1.0 - total_penalty)

    return CFSResult(
        score=score,
        violations=all_violations,
        total_penalty=total_penalty,
        block_count=len(schedule),
        days_evaluated=len(days),
    )


# ============================================================================
# BATCH EVALUATION (for use with generated scenarios)
# ============================================================================

def evaluate_scenarios_file(scenarios_path: str) -> list[dict]:
    """Load scenarios JSON and compute CFS for each."""
    import json

    with open(scenarios_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for s_data in data["scenarios"]:
        profile = ADHDProfile.model_validate(s_data["profile"])
        schedule = [ScheduleBlock.model_validate(b) for b in s_data["schedule"]]

        cfs = cognitive_feasibility_score(schedule, profile)
        results.append({
            "scenario_id": s_data["scenario_id"],
            "scenario_type": s_data["scenario_type"],
            "profile_id": profile.profile_id,
            "cfs_score": round(cfs.score, 4),
            "violation_count": cfs.violation_count,
            "total_penalty": round(cfs.total_penalty, 4),
            "violations_by_rule": _count_by_rule(cfs.violations),
        })

    return results


def _count_by_rule(violations: list[Violation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in violations:
        counts[v.rule] = counts.get(v.rule, 0) + 1
    return counts


if __name__ == "__main__":
    import json

    scenarios_path = PROJECT_ROOT / "research" / "scenarios" / "scenarios_50.json"
    if not scenarios_path.exists():
        print(f"No scenarios file at {scenarios_path}. Run generate_scenarios.py first.")
        sys.exit(1)

    results = evaluate_scenarios_file(str(scenarios_path))

    print(f"\nCFS Evaluation — {len(results)} scenarios")
    print("=" * 60)

    scores = [r["cfs_score"] for r in results]
    print(f"Mean CFS:   {sum(scores)/len(scores):.3f}")
    print(f"Min CFS:    {min(scores):.3f}")
    print(f"Max CFS:    {max(scores):.3f}")

    print(f"\nPer-scenario results:")
    for r in sorted(results, key=lambda x: x["cfs_score"]):
        print(f"  {r['scenario_id']:<50} CFS={r['cfs_score']:.3f}  violations={r['violation_count']}")

    # Save evaluation results
    eval_path = PROJECT_ROOT / "research" / "scenarios" / "cfs_evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation to {eval_path}")
