"""
Generate 50 Synthetic ADHD Scheduling Scenarios.

10 scenario types × 5 ADHD profile archetypes = 50 total.
Uses OpenAI API (gpt-5.4) to generate realistic weekly schedules,
then validates them against the ADHD constraint ontology.

Usage:
    python -m research.data.generate_scenarios [--model gpt-5.4] [--output research/scenarios/scenarios_50.json]

Output: JSON file with 50 Scenario objects ready for CFS evaluation.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.ontology.adhd_constraints import (
    ADHD_ARCHETYPES,
    SCENARIO_TYPE_DESCRIPTIONS,
    ADHDProfile,
    Scenario,
    ScenarioType,
    ScheduleBlock,
)

load_dotenv(PROJECT_ROOT / ".env")

# ============================================================================
# GENERATION PROMPT
# ============================================================================

SYSTEM_PROMPT = """\
You are an expert in ADHD, cognitive load theory, and academic scheduling.
You generate realistic synthetic scheduling scenarios for ADHD students.

IMPORTANT RULES:
1. Generate a COMPLETE weekly schedule (Monday-Friday, optionally Saturday/Sunday).
2. Include BOTH fixed obligations (classes, labs, work) AND flexible study/task blocks.
3. Use 24-hour time format (HH:MM).
4. Each schedule block must have a cognitive_load category: "low", "medium", or "high".
5. task_type must be one of: lecture, study, exam, lab, group_work, break, exercise, social, admin, creative.
6. Schedules must be REALISTIC — no 14-hour study days, include meals/breaks.
7. Include at least 12 and at most 35 schedule blocks.
8. The schedule should INTENTIONALLY contain ADHD-unfriendly patterns that a smart scheduler would need to fix (e.g., high-load tasks during trough hours, missing breaks, etc.)
9. expected_challenges should list 3-5 specific ADHD-related issues with this schedule.
10. optimal_interventions should list 2-4 things an ADHD-aware scheduler would do differently.

Respond with ONLY valid JSON matching the schema below. No markdown, no explanation."""


def build_user_prompt(profile: ADHDProfile, scenario_type: ScenarioType) -> str:
    desc = SCENARIO_TYPE_DESCRIPTIONS[scenario_type]
    profile_json = profile.model_dump_json(indent=2)

    return f"""\
Generate a synthetic weekly scheduling scenario.

SCENARIO TYPE: {scenario_type.value}
DESCRIPTION: {desc}

STUDENT ADHD PROFILE:
{profile_json}

Generate a JSON object with this exact structure:
{{
  "scenario_id": "{profile.profile_id}__{scenario_type.value}",
  "scenario_type": "{scenario_type.value}",
  "week_context": "<2-3 sentence narrative about this student's week>",
  "schedule": [
    {{
      "title": "<descriptive title>",
      "day": "<monday|tuesday|wednesday|thursday|friday|saturday|sunday>",
      "start_time": "HH:MM",
      "end_time": "HH:MM",
      "cognitive_load": "<low|medium|high>",
      "task_type": "<lecture|study|exam|lab|group_work|break|exercise|social|admin|creative>",
      "course": "<course code or null>",
      "is_fixed": <true for lectures/exams/labs, false for flexible blocks>,
      "is_decomposed": false,
      "notes": "<optional note or null>"
    }}
  ],
  "expected_challenges": ["<challenge 1>", "<challenge 2>", ...],
  "optimal_interventions": ["<intervention 1>", "<intervention 2>", ...]
}}

IMPORTANT: The schedule must reflect BOTH the scenario type constraints AND the student's ADHD profile.
Include realistic course names (e.g., CS301, PSYCH101, MATH220).
The schedule should contain some ADHD-unfriendly patterns that need fixing."""


# ============================================================================
# GENERATION ENGINE
# ============================================================================

def generate_single_scenario(
    client: OpenAI,
    model: str,
    profile: ADHDProfile,
    scenario_type: ScenarioType,
    max_retries: int = 2,
) -> Scenario | None:
    """Generate one scenario via OpenAI API with validation + retry."""

    user_prompt = build_user_prompt(profile, scenario_type)
    scenario_id = f"{profile.profile_id}__{scenario_type.value}"

    for attempt in range(max_retries + 1):
        try:
            print(f"  [{attempt+1}/{max_retries+1}] Generating {scenario_id}...")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,
                max_completion_tokens=4096,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[: raw.rfind("```")]
                raw = raw.strip()

            data = json.loads(raw)

            # Inject the profile back in (LLM only generated schedule + context)
            data["profile"] = profile.model_dump()
            data["scenario_id"] = scenario_id
            data["scenario_type"] = scenario_type.value

            scenario = Scenario.model_validate(data)

            block_count = len(scenario.schedule)
            if block_count < 8:
                print(f"    Warning: only {block_count} blocks, retrying...")
                continue

            print(f"    OK — {block_count} blocks, {len(scenario.expected_challenges)} challenges")
            return scenario

        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e}")
        except ValidationError as e:
            print(f"    Validation error: {e.error_count()} issues")
            for err in e.errors()[:3]:
                print(f"      - {err['loc']}: {err['msg']}")
        except Exception as e:
            print(f"    Unexpected error: {e}")

        if attempt < max_retries:
            time.sleep(2)

    print(f"    FAILED after {max_retries + 1} attempts: {scenario_id}")
    return None


def generate_all_scenarios(
    model: str = "gpt-5.4",
    output_path: str = "research/scenarios/scenarios_50.json",
) -> list[Scenario]:
    """Generate all 50 scenarios (10 types × 5 profiles)."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or .env file")

    client = OpenAI(api_key=api_key)

    scenarios: list[Scenario] = []
    failed: list[str] = []
    total = len(ADHD_ARCHETYPES) * len(ScenarioType)

    print(f"Generating {total} scenarios with model={model}")
    print(f"Profiles: {len(ADHD_ARCHETYPES)}, Scenario types: {len(ScenarioType)}")
    print("=" * 60)

    for i, scenario_type in enumerate(ScenarioType):
        print(f"\n[{i+1}/{len(ScenarioType)}] Scenario type: {scenario_type.value}")
        print(f"  {SCENARIO_TYPE_DESCRIPTIONS[scenario_type][:80]}...")

        for profile in ADHD_ARCHETYPES:
            result = generate_single_scenario(client, model, profile, scenario_type)
            if result:
                scenarios.append(result)
            else:
                failed.append(f"{profile.profile_id}__{scenario_type.value}")

            # Rate limiting — be respectful
            time.sleep(1)

    # Save results
    out_path = PROJECT_ROOT / output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "total_generated": len(scenarios),
            "total_failed": len(failed),
            "model": model,
            "generation_note": "Synthetic scenarios generated by LLM, reviewed manually. "
                               "Not a clinically validated dataset.",
            "profile_count": len(ADHD_ARCHETYPES),
            "scenario_type_count": len(ScenarioType),
        },
        "failed_ids": failed,
        "scenarios": [s.model_dump() for s in scenarios],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(scenarios)}/{total} scenarios generated.")
    print(f"Failed: {len(failed)}")
    if failed:
        for fid in failed:
            print(f"  - {fid}")
    print(f"Output: {out_path}")

    return scenarios


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate 50 synthetic ADHD scheduling scenarios")
    parser.add_argument(
        "--model", default="gpt-5.4",
        help="OpenAI model to use (default: gpt-5.4). Try gpt-4o as cheaper fallback.",
    )
    parser.add_argument(
        "--output", default="research/scenarios/scenarios_50.json",
        help="Output JSON path (relative to project root)",
    )
    args = parser.parse_args()

    generate_all_scenarios(model=args.model, output_path=args.output)


if __name__ == "__main__":
    main()
