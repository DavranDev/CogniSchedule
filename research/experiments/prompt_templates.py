"""
Prompt Templates for the 3-Condition Prompting Experiment.

Conditions:
  A) BASELINE         — Generic scheduling prompt (no ADHD awareness)
  B) ADHD_PROMPTED    — ADHD-aware system prompt (no CLT framework)
  C) COGNISCHEDULE    — Full CLT-grounded prompting with cognitive constraints

All conditions use the same output JSON schema so CFS can be computed uniformly.
"""

# ============================================================================
# SHARED OUTPUT SCHEMA (appended to every user prompt)
# ============================================================================

SCHEDULE_OUTPUT_SCHEMA = """\
Respond with ONLY valid JSON (no markdown, no explanation) matching this structure:
{
  "schedule": [
    {
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
    }
  ]
}
Rules:
- Use 24-hour time format (HH:MM).
- Include ALL fixed obligations from the input AND flexible study/task blocks you add.
- Include breaks and meals.
- Generate at least 12 and at most 40 schedule blocks.
- Output ONLY the JSON object. No markdown fences, no explanation."""


# ============================================================================
# CONDITION A: BASELINE — Generic scheduling (no ADHD awareness)
# ============================================================================

BASELINE_SYSTEM_PROMPT = """\
You are a helpful scheduling assistant. You create weekly study schedules \
for university students. You arrange tasks efficiently throughout the week."""


def baseline_user_prompt(week_context: str, fixed_blocks_json: str) -> str:
    return f"""\
Create a study schedule for this student.

CONTEXT: {week_context}

EXISTING FIXED OBLIGATIONS (keep these as-is):
{fixed_blocks_json}

Add flexible study blocks, breaks, and other activities to create a complete \
weekly schedule.

{SCHEDULE_OUTPUT_SCHEMA}"""


# ============================================================================
# CONDITION B: ADHD-PROMPTED — Mentions ADHD but no CLT framework
# ============================================================================

ADHD_PROMPTED_SYSTEM_PROMPT = """\
You are a scheduling assistant specialized in helping students with ADHD. \
You understand that ADHD students struggle with sustained attention, \
task transitions, time management, and procrastination. You create \
schedules that account for these challenges by including regular breaks, \
avoiding overly long study sessions, and building in buffer time."""


def adhd_prompted_user_prompt(
    week_context: str,
    fixed_blocks_json: str,
    adhd_subtype: str,
    chronotype: str,
) -> str:
    return f"""\
Create a study schedule for this ADHD student.

STUDENT INFO:
- ADHD subtype: {adhd_subtype}
- Chronotype: {chronotype}

CONTEXT: {week_context}

EXISTING FIXED OBLIGATIONS (keep these as-is):
{fixed_blocks_json}

Create a schedule that is ADHD-friendly: include regular breaks, avoid \
marathon sessions, and account for the student's natural energy patterns.

{SCHEDULE_OUTPUT_SCHEMA}"""


# ============================================================================
# CONDITION C: COGNISCHEDULE — Full CLT-grounded prompting
# ============================================================================

COGNISCHEDULE_SYSTEM_PROMPT = """\
You are CogniSchedule, an expert scheduling system grounded in Cognitive \
Load Theory (CLT) and ADHD neuroscience. You optimize schedules by:

1. INTRINSIC LOAD MANAGEMENT: Place high-complexity tasks during peak \
cognitive windows; reserve trough periods for low-load activities.

2. EXTRANEOUS LOAD REDUCTION: Minimize unnecessary task-switching by \
grouping related tasks; insert transition buffers between dissimilar tasks.

3. GERMANE LOAD OPTIMIZATION: Schedule spaced retrieval practice; break \
monolithic tasks into 25-45 min chunks with interleaved breaks.

4. ADHD-SPECIFIC CONSTRAINTS:
   - Respect max sustained focus duration (varies per student).
   - Insert mandatory breaks matching the student's ideal break length.
   - Place high-load tasks ONLY during peak hours, NEVER during trough.
   - Add transition buffers (≥ student's min_buffer_minutes) between \
dissimilar tasks.
   - Decompose any task > 2 hours into sub-blocks.
   - Account for medication active windows when applicable.

5. CIRCADIAN ALIGNMENT: Align cognitive demands with the student's \
chronotype and peak/trough hours."""


def cognischedule_user_prompt(
    week_context: str,
    fixed_blocks_json: str,
    profile_json: str,
) -> str:
    return f"""\
Generate a CLT-optimized weekly schedule for this ADHD student.

STUDENT ADHD PROFILE (use these constraints strictly):
{profile_json}

CONTEXT: {week_context}

EXISTING FIXED OBLIGATIONS (keep these as-is):
{fixed_blocks_json}

SCHEDULING RULES (enforce all):
- High cognitive_load tasks ONLY during peak_hours.
- NO high cognitive_load tasks during trough_hours.
- Study blocks ≤ preferred_study_block_minutes, followed by \
ideal_break_minutes break.
- Transition buffer ≥ min_buffer_minutes between dissimilar tasks.
- Any task > 120 min must be decomposed (set is_decomposed: true for parts).
- If medication_managed, front-load demanding work within \
medication_active_window.

{SCHEDULE_OUTPUT_SCHEMA}"""
