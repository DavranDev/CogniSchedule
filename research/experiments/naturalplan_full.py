"""
Full NaturalPlan Calendar Scheduling benchmark evaluation (1,000 examples).

Features:
- Runs GPT-OSS-120B on all 1,000 examples (0-shot, temperature=0)
- Parallel execution via ThreadPoolExecutor
- Dual scoring: original NaturalPlan regex + our LLM extraction pipeline
- Constraint validation for "incorrect" predictions
- Incremental saving after each batch
- Wilson 95% CIs on all accuracy numbers
- Generates Table 1 JSON artifact

Usage:
    python -m research.experiments.naturalplan_full --parallel 10
    python -m research.experiments.naturalplan_full --analyze   # analysis only (no API calls)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_CSV = PROJECT_ROOT / "natural_plan_experiments" / "exp_multiday_or_multipeople" / "calendar_scheduling.csv"
RESULTS_DIR = PROJECT_ROOT / "research" / "experiments" / "results"
RAW_RESULTS_FILE = RESULTS_DIR / "naturalplan_full_raw.json"
ANALYSIS_FILE = RESULTS_DIR / "table1_naturalplan.json"

# ── Model config ─────────────────────────────────────────────────────────────
GENERATOR_MODEL = "openai/gpt-oss-120b"
HELPER_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Regex / Parsing (reused from original pipeline) ─────────────────────────
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DAY_PATTERN = r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
TIME_PATTERN = r"(?:[01]?\d|2[0-3]):[0-5]\d|(?:1[0-2]|0?[1-9])(?::[0-5]\d)?\s?(?:am|pm|AM|PM)"
SLOT_PATTERN = rf"(?P<day>{DAY_PATTERN})\s*(?:,|\||\s+from|\s+at)?\s*(?P<start>{TIME_PATTERN})\s*(?:-|–|—|to)\s*(?P<end>{TIME_PATTERN})"
GENERIC_SLOT_RE = re.compile(SLOT_PATTERN, re.IGNORECASE)

# Original NaturalPlan regex (from google-deepmind/natural-plan)
ORIGINAL_NP_RE = re.compile(r"[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+")


def safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def parse_time_to_minutes(raw_time: str) -> Optional[int]:
    token = raw_time.strip().lower().replace(" ", "")
    if re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", token):
        hour, minute = token.split(":")
        return int(hour) * 60 + int(minute)
    match_12h = re.fullmatch(r"(1[0-2]|0?[1-9])(?::([0-5]\d))?(am|pm)", token)
    if match_12h:
        hour = int(match_12h.group(1))
        minute = int(match_12h.group(2) or 0)
        suffix = match_12h.group(3)
        if suffix == "am":
            hour = 0 if hour == 12 else hour
        else:
            hour = hour if hour == 12 else hour + 12
        return hour * 60 + minute
    return None


def format_minutes_as_time(minutes: int) -> str:
    return f"{minutes // 60}:{minutes % 60:02d}"


def normalize_slot(day: str, start_minutes: int, end_minutes: int) -> str:
    return f"Here is the proposed time: {day.capitalize()}, {format_minutes_as_time(start_minutes)} - {format_minutes_as_time(end_minutes)}"


def extract_slot_regex(text: str) -> Optional[Tuple[str, int, int]]:
    if not text:
        return None
    match = GENERIC_SLOT_RE.search(text)
    if match:
        day, start, end = match.group("day"), match.group("start"), match.group("end")
        start_min = parse_time_to_minutes(start)
        end_min = parse_time_to_minutes(end)
        if start_min is not None and end_min is not None:
            return day, start_min, end_min
    return None


def normalize_formatted_answer(text: str) -> str:
    slot = extract_slot_regex(text)
    if not slot:
        return "No valid time found"
    return normalize_slot(*slot)


# ── Original NaturalPlan evaluation (exact reimplementation) ─────────────────

def original_naturalplan_eval(model_answer: str, golden_answer: str) -> bool:
    """Evaluate using the original NaturalPlan regex (Zheng et al.).

    The original evaluator:
    1. Extracts all matches of pattern [Day, HH:MM - HH:MM] from the response
    2. Takes the LAST match (final answer)
    3. Compares string equality with golden answer
    """
    if not model_answer or not golden_answer:
        return False

    # Extract matches from model answer
    model_matches = ORIGINAL_NP_RE.findall(model_answer)
    if not model_matches:
        return False

    # Take the last match (original NaturalPlan behavior)
    model_slot = model_matches[-1].strip()

    # Extract from golden answer
    golden_matches = ORIGINAL_NP_RE.findall(golden_answer)
    if not golden_matches:
        return False
    golden_slot = golden_matches[-1].strip()

    return model_slot == golden_slot


# ── LLM-based evaluation pipeline ───────────────────────────────────────────

def get_llm_response(client, prompt: str, model: str, max_tokens: int = 256) -> str:
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0,
            max_tokens=max_tokens,
        )
        content = completion.choices[0].message.content
        return content.strip() if content else ""
    except Exception as exc:
        return f"Error: {exc}"


def helper_extract(client, raw_text: str, golden_answer: str) -> str:
    if "Error" in raw_text:
        return raw_text
    extraction_prompt = f"""You are a strict data formatter.

Goal:
- Extract one proposed meeting slot from the GPT answer.
- Output it in the same style as the golden answer format.

Golden answer format example:
"{golden_answer}"

Required output format:
"Here is the proposed time: [Day], [Start Time] - [End Time]"

Rules:
1. Return only one line and no explanation.
2. Convert 12-hour time to 24-hour time.
3. If no valid slot exists, return exactly: "No valid time found"

GPT answer:
"{raw_text}"

Output:
"""
    return get_llm_response(client, extraction_prompt, model=HELPER_MODEL, max_tokens=100)


def llm_evaluate_row(client, golden_answer: str, model_answer: str) -> Tuple[str, bool, str]:
    normalized_golden = normalize_formatted_answer(golden_answer)
    extracted_raw = helper_extract(client, model_answer, golden_answer)
    if extracted_raw.startswith("Error:"):
        fallback_extracted = normalize_formatted_answer(model_answer)
        is_correct = fallback_extracted != "No valid time found" and fallback_extracted == normalized_golden
        return fallback_extracted, is_correct, "deterministic_fallback"
    normalized_extracted = normalize_formatted_answer(extracted_raw)
    is_correct = normalized_extracted != "No valid time found" and normalized_extracted == normalized_golden
    return normalized_extracted, is_correct, "llm_helper_extract"


# ── Constraint validation ────────────────────────────────────────────────────

def parse_busy_times_from_prompt(prompt: str) -> dict[str, list[Tuple[int, int]]]:
    """Parse busy time blocks from the NaturalPlan prompt for each person per day.

    Handles NaturalPlan formats:
    - "Michelle has meetings on Monday during 11:00 to 12:00"
    - "Steven has blocked their calendar on Monday during 9:00 to 9:30, 11:30 to 12:00"
    - "Jerry is busy on Monday from 9:00 to 10:00"

    Returns: {person_day: [(start_min, end_min), ...]}
    """
    busy = {}
    TIME_24 = r"\d{1,2}:\d{2}"

    # Pattern: "Person has meetings/blocked ... on Day during TIME to TIME[, TIME to TIME]*"
    line_pattern = re.compile(
        rf"(\w[\w ]*?)\s+(?:has\s+(?:meetings|blocked\s+their\s+calendar)|is\s+busy)\s+"
        rf"(?:on\s+)?({DAY_PATTERN})\s+(?:during|from)\s+"
        rf"((?:{TIME_24}\s*(?:to|-|–)\s*{TIME_24}(?:\s*,\s*)?)+)",
        re.IGNORECASE
    )
    slot_pattern = re.compile(rf"({TIME_24})\s*(?:to|-|–)\s*({TIME_24})")

    for line_match in line_pattern.finditer(prompt):
        person = line_match.group(1).strip()
        day = line_match.group(2).capitalize()
        slots_str = line_match.group(3)
        key = f"{person}_{day}"

        for slot_match in slot_pattern.finditer(slots_str):
            start = parse_time_to_minutes(slot_match.group(1))
            end = parse_time_to_minutes(slot_match.group(2))
            if start is not None and end is not None:
                busy.setdefault(key, []).append((start, end))

    return busy


def parse_duration_from_prompt(prompt: str) -> Optional[int]:
    """Extract meeting duration from prompt."""
    # "for half an hour" or "for 30 minutes" or "for an hour" or "for one hour" or "for 60 minutes"
    if re.search(r"half\s+an?\s+hour|30\s*(?:minute|min)", prompt, re.IGNORECASE):
        return 30
    if re.search(r"(?:an|one|1)\s+hour|60\s*(?:minute|min)", prompt, re.IGNORECASE):
        return 60
    return None


def parse_people_from_prompt(prompt: str) -> list[str]:
    """Extract participant names from the prompt.

    NaturalPlan format: "schedule a meeting for Michelle, Steven and Jerry for one hour"
    """
    m = re.search(r"schedule\s+a\s+meeting\s+for\s+([\w\s,]+?)\s+for\s+", prompt, re.IGNORECASE)
    if m:
        names_str = m.group(1)
        names = re.split(r",\s*(?:and\s+)?|\s+and\s+", names_str)
        return [n.strip() for n in names if n.strip()]
    return []


def parse_work_hours_from_prompt(prompt: str) -> Tuple[int, int]:
    """Extract work hours from prompt (default 9:00-17:00)."""
    m = re.search(r"work\s+hours?\s+of\s+(\d{1,2}:\d{2})\s*to\s*(\d{1,2}:\d{2})", prompt, re.IGNORECASE)
    if m:
        start = parse_time_to_minutes(m.group(1))
        end = parse_time_to_minutes(m.group(2))
        if start is not None and end is not None:
            return start, end
    return 540, 1020  # 9:00-17:00


def validate_slot_against_constraints(
    prompt: str,
    proposed_day: str,
    proposed_start: int,
    proposed_end: int,
) -> Tuple[bool, str]:
    """Check if a proposed slot violates any constraints in the prompt.

    Returns: (is_valid, reason)
    """
    busy = parse_busy_times_from_prompt(prompt)
    duration = parse_duration_from_prompt(prompt)
    people = parse_people_from_prompt(prompt)
    work_start, work_end = parse_work_hours_from_prompt(prompt)

    # Check work hours
    if proposed_start < work_start or proposed_end > work_end:
        return False, "outside_work_hours"

    # Check duration
    if duration is not None:
        actual_duration = proposed_end - proposed_start
        if actual_duration != duration:
            return False, f"wrong_duration (expected {duration}, got {actual_duration})"

    # Check conflicts with busy times for all participants on the proposed day
    for key, slots in busy.items():
        if proposed_day in key:
            for busy_start, busy_end in slots:
                if proposed_start < busy_end and proposed_end > busy_start:
                    return False, f"conflict: {key} ({format_minutes_as_time(busy_start)}-{format_minutes_as_time(busy_end)})"

    return True, "valid"


# ── Statistics ───────────────────────────────────────────────────────────────

def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denom
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * total)) / total) / denom
    return max(0, center - margin), min(1, center + margin)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df["num_people"] = df["num_people"].astype(int)
    df["num_days"] = df["num_days"].astype(int)
    print(f"Loaded {len(df)} examples from {path}")
    print(f"  num_people: {df['num_people'].min()}-{df['num_people'].max()}")
    print(f"  num_days: {df['num_days'].min()}-{df['num_days'].max()}")
    return df


# ── Single trial ─────────────────────────────────────────────────────────────

save_lock = Lock()


def run_single_example(
    example: dict,
    client,
    idx: int,
    total: int,
) -> dict:
    """Run one example: generate answer, dual-score, constraint-validate."""
    example_id = example["example_id"]
    prompt = safe_text(example.get("prompt_0shot", ""))
    golden_answer = safe_text(example.get("golden_plan", ""))
    num_people = int(example.get("num_people", 0))
    num_days = int(example.get("num_days", 0))

    # Generate model answer
    full_prompt = f"{prompt}\nSOLUTION:"
    model_answer = get_llm_response(client, full_prompt, model=GENERATOR_MODEL, max_tokens=65536)

    # Score 1: Original NaturalPlan regex
    original_correct = original_naturalplan_eval(model_answer, golden_answer)

    # Score 2: LLM extraction pipeline
    extracted_answer, llm_correct, judge_source = llm_evaluate_row(client, golden_answer, model_answer)

    # Constraint validation (if LLM eval says incorrect)
    constraint_valid = None
    constraint_reason = None
    if not llm_correct:
        slot = extract_slot_regex(extracted_answer)
        if slot:
            day, start_min, end_min = slot
            constraint_valid, constraint_reason = validate_slot_against_constraints(
                prompt, day, start_min, end_min
            )
        else:
            constraint_valid = False
            constraint_reason = "no_slot_extracted"

    status = "✓" if llm_correct else ("~" if constraint_valid else "✗")
    print(f"  [{idx+1:4d}/{total}] {example_id}: orig={'✓' if original_correct else '✗'} llm={status}  (p={num_people}, d={num_days})")

    return {
        "example_id": example_id,
        "num_people": num_people,
        "num_days": num_days,
        "golden_answer": golden_answer,
        "model_answer": model_answer,
        "extracted_answer": extracted_answer,
        "judge_source": judge_source,
        "original_regex_correct": original_correct,
        "llm_extraction_correct": llm_correct,
        "constraint_valid": constraint_valid,
        "constraint_reason": constraint_reason,
    }


# ── Main execution ───────────────────────────────────────────────────────────

def run_experiment(parallel: int = 1) -> list[dict]:
    """Run GPT-OSS-120B on all 1,000 NaturalPlan examples."""
    from groq import Groq

    df = load_dataset(DATASET_CSV)
    examples = df.to_dict(orient="records")
    total = len(examples)

    # Load existing results for resume
    existing_results = []
    done_ids = set()
    if RAW_RESULTS_FILE.exists():
        with open(RAW_RESULTS_FILE, "r") as f:
            existing_results = json.load(f)
        done_ids = {r["example_id"] for r in existing_results}
        print(f"Resuming: {len(done_ids)} already completed, {total - len(done_ids)} remaining")

    remaining = [ex for ex in examples if ex["example_id"] not in done_ids]
    if not remaining:
        print("All examples already completed!")
        return existing_results

    results = list(existing_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def save_incremental():
        with save_lock:
            with open(RAW_RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

    if parallel <= 1:
        client = Groq(api_key=GROQ_API_KEY)
        for i, example in enumerate(remaining):
            result = run_single_example(example, client, len(done_ids) + i, total)
            results.append(result)
            if (i + 1) % 10 == 0:
                save_incremental()
                print(f"  [saved {len(results)}/{total}]")
    else:
        # Parallel execution — one Groq client per thread
        def worker(args):
            idx, example = args
            client = Groq(api_key=GROQ_API_KEY)
            return run_single_example(example, client, idx, total)

        batch_size = parallel * 2
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            indexed_batch = [(len(done_ids) + batch_start + i, ex) for i, ex in enumerate(batch)]

            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {executor.submit(worker, args): args for args in indexed_batch}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        with save_lock:
                            results.append(result)
                    except Exception as exc:
                        idx, example = futures[future]
                        print(f"  ERROR on {example['example_id']}: {exc}")
                        with save_lock:
                            results.append({
                                "example_id": example["example_id"],
                                "num_people": int(example.get("num_people", 0)),
                                "num_days": int(example.get("num_days", 0)),
                                "golden_answer": safe_text(example.get("golden_plan", "")),
                                "model_answer": f"Error: {exc}",
                                "extracted_answer": "",
                                "judge_source": "error",
                                "original_regex_correct": False,
                                "llm_extraction_correct": False,
                                "constraint_valid": False,
                                "constraint_reason": f"error: {exc}",
                            })

            save_incremental()
            print(f"  [saved {len(results)}/{total}]")

    save_incremental()
    print(f"\nExperiment complete: {len(results)} results saved to {RAW_RESULTS_FILE}")
    return results


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_results(results: list[dict]) -> dict:
    """Generate Table 1 analysis from raw results."""
    n = len(results)
    print(f"\n{'='*60}")
    print(f"  NaturalPlan Full Benchmark Analysis (N={n})")
    print(f"{'='*60}")

    # Overall accuracy — dual scoring
    orig_correct = sum(1 for r in results if r["original_regex_correct"])
    llm_correct = sum(1 for r in results if r["llm_extraction_correct"])

    # Constraint validation
    incorrect_by_llm = [r for r in results if not r["llm_extraction_correct"]]
    constraint_valid_count = sum(1 for r in incorrect_by_llm if r.get("constraint_valid"))
    actual_violations = sum(1 for r in incorrect_by_llm if not r.get("constraint_valid"))
    constraint_validated_correct = llm_correct + constraint_valid_count

    orig_ci = wilson_interval(orig_correct, n)
    llm_ci = wilson_interval(llm_correct, n)
    cv_ci = wilson_interval(constraint_validated_correct, n)

    print(f"\n── Overall Accuracy ──")
    print(f"  Original regex:        {orig_correct}/{n} = {orig_correct/n*100:.1f}% [{orig_ci[0]*100:.1f}, {orig_ci[1]*100:.1f}]")
    print(f"  LLM extraction:        {llm_correct}/{n} = {llm_correct/n*100:.1f}% [{llm_ci[0]*100:.1f}, {llm_ci[1]*100:.1f}]")
    print(f"  Constraint-validated:   {constraint_validated_correct}/{n} = {constraint_validated_correct/n*100:.1f}% [{cv_ci[0]*100:.1f}, {cv_ci[1]*100:.1f}]")
    print(f"\n  Of {len(incorrect_by_llm)} LLM-incorrect: {constraint_valid_count} valid alternatives, {actual_violations} true violations")

    # Breakdown by num_days
    print(f"\n── By Number of Days ──")
    by_days = {}
    for r in results:
        d = r["num_days"]
        by_days.setdefault(d, []).append(r)

    days_breakdown = {}
    for d in sorted(by_days.keys()):
        group = by_days[d]
        gn = len(group)
        g_orig = sum(1 for r in group if r["original_regex_correct"])
        g_llm = sum(1 for r in group if r["llm_extraction_correct"])
        g_inc = [r for r in group if not r["llm_extraction_correct"]]
        g_cv = g_llm + sum(1 for r in g_inc if r.get("constraint_valid"))
        g_orig_ci = wilson_interval(g_orig, gn)
        g_llm_ci = wilson_interval(g_llm, gn)
        g_cv_ci = wilson_interval(g_cv, gn)
        print(f"  {d} day(s) (n={gn}): orig={g_orig/gn*100:.1f}%, llm={g_llm/gn*100:.1f}%, cv={g_cv/gn*100:.1f}%")
        days_breakdown[str(d)] = {
            "n": gn,
            "original_regex": {"correct": g_orig, "accuracy": round(g_orig/gn*100, 1), "ci95": [round(g_orig_ci[0]*100, 1), round(g_orig_ci[1]*100, 1)]},
            "llm_extraction": {"correct": g_llm, "accuracy": round(g_llm/gn*100, 1), "ci95": [round(g_llm_ci[0]*100, 1), round(g_llm_ci[1]*100, 1)]},
            "constraint_validated": {"correct": g_cv, "accuracy": round(g_cv/gn*100, 1), "ci95": [round(g_cv_ci[0]*100, 1), round(g_cv_ci[1]*100, 1)]},
        }

    # Breakdown by num_people
    print(f"\n── By Number of People ──")
    by_people = {}
    for r in results:
        p = r["num_people"]
        by_people.setdefault(p, []).append(r)

    people_breakdown = {}
    for p in sorted(by_people.keys()):
        group = by_people[p]
        gn = len(group)
        g_orig = sum(1 for r in group if r["original_regex_correct"])
        g_llm = sum(1 for r in group if r["llm_extraction_correct"])
        g_inc = [r for r in group if not r["llm_extraction_correct"]]
        g_cv = g_llm + sum(1 for r in g_inc if r.get("constraint_valid"))
        g_orig_ci = wilson_interval(g_orig, gn)
        g_llm_ci = wilson_interval(g_llm, gn)
        g_cv_ci = wilson_interval(g_cv, gn)
        print(f"  {p} people (n={gn}): orig={g_orig/gn*100:.1f}%, llm={g_llm/gn*100:.1f}%, cv={g_cv/gn*100:.1f}%")
        people_breakdown[str(p)] = {
            "n": gn,
            "original_regex": {"correct": g_orig, "accuracy": round(g_orig/gn*100, 1), "ci95": [round(g_orig_ci[0]*100, 1), round(g_orig_ci[1]*100, 1)]},
            "llm_extraction": {"correct": g_llm, "accuracy": round(g_llm/gn*100, 1), "ci95": [round(g_llm_ci[0]*100, 1), round(g_llm_ci[1]*100, 1)]},
            "constraint_validated": {"correct": g_cv, "accuracy": round(g_cv/gn*100, 1), "ci95": [round(g_cv_ci[0]*100, 1), round(g_cv_ci[1]*100, 1)]},
        }

    # Error categorization
    print(f"\n── Error Analysis (LLM-incorrect examples) ──")
    error_cats = {"valid_alternative": 0, "constraint_violation": 0, "no_slot_extracted": 0, "error": 0}
    for r in incorrect_by_llm:
        reason = r.get("constraint_reason", "")
        if r.get("constraint_valid"):
            error_cats["valid_alternative"] += 1
        elif reason == "no_slot_extracted":
            error_cats["no_slot_extracted"] += 1
        elif reason and reason.startswith("error"):
            error_cats["error"] += 1
        else:
            error_cats["constraint_violation"] += 1

    for cat, count in error_cats.items():
        pct = count / len(incorrect_by_llm) * 100 if incorrect_by_llm else 0
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # Reference: Original NaturalPlan results
    print(f"\n── Reference Baselines ──")
    print(f"  Gemini 1.5 Pro (Zheng et al., 5-shot, original regex): 48.9%")
    print(f"  GPT-OSS-120B (this work, 0-shot, original regex):      {orig_correct/n*100:.1f}%")
    print(f"  GPT-OSS-120B (this work, 0-shot, LLM extraction):      {llm_correct/n*100:.1f}%")
    print(f"  GPT-OSS-120B (this work, 0-shot, constraint-valid):    {constraint_validated_correct/n*100:.1f}%")

    # Build Table 1 JSON
    table1 = {
        "table_name": "Table 1: NaturalPlan Calendar Scheduling (Full Benchmark)",
        "model": GENERATOR_MODEL,
        "n_examples": n,
        "prompting": "0-shot",
        "temperature": 0,
        "overall": {
            "original_regex": {
                "correct": orig_correct,
                "accuracy_pct": round(orig_correct / n * 100, 1),
                "ci95_pct": [round(orig_ci[0] * 100, 1), round(orig_ci[1] * 100, 1)],
            },
            "llm_extraction": {
                "correct": llm_correct,
                "accuracy_pct": round(llm_correct / n * 100, 1),
                "ci95_pct": [round(llm_ci[0] * 100, 1), round(llm_ci[1] * 100, 1)],
            },
            "constraint_validated": {
                "correct": constraint_validated_correct,
                "accuracy_pct": round(constraint_validated_correct / n * 100, 1),
                "ci95_pct": [round(cv_ci[0] * 100, 1), round(cv_ci[1] * 100, 1)],
            },
        },
        "error_analysis": {
            "total_llm_incorrect": len(incorrect_by_llm),
            "valid_alternatives": constraint_valid_count,
            "true_violations": error_cats["constraint_violation"],
            "no_slot_extracted": error_cats["no_slot_extracted"],
            "errors": error_cats["error"],
            "valid_alternative_pct": round(constraint_valid_count / len(incorrect_by_llm) * 100, 1) if incorrect_by_llm else 0,
        },
        "by_num_days": days_breakdown,
        "by_num_people": people_breakdown,
        "reference_baselines": {
            "gemini_1_5_pro_zheng_5shot": {"accuracy_pct": 48.9, "evaluator": "original_regex", "n": 1000},
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_FILE, "w") as f:
        json.dump(table1, f, indent=2)
    print(f"\nTable 1 saved to {ANALYSIS_FILE}")

    return table1


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--analyze", action="store_true", help="Run analysis only (no API calls)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.analyze:
        if not RAW_RESULTS_FILE.exists():
            print(f"No raw results found at {RAW_RESULTS_FILE}. Run the experiment first.")
            sys.exit(1)
        with open(RAW_RESULTS_FILE, "r") as f:
            results = json.load(f)
        analyze_results(results)
    else:
        results = run_experiment(parallel=args.parallel)
        analyze_results(results)


if __name__ == "__main__":
    main()
