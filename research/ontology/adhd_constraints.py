"""
ADHD Constraint Ontology — Formal specification of ADHD scheduling constraints.

Defines Pydantic models for:
  - ADHD student profiles (5 archetypes)
  - Scheduling blocks with cognitive load categories
  - Scenario specifications (10 types × 5 profiles = 50)

These models are used by the scenario generator, the CFS metric,
and the neuro-symbolic scheduler.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================================
# ENUMS
# ============================================================================

class ADHDSubtype(str, Enum):
    INATTENTIVE = "inattentive"
    HYPERACTIVE = "hyperactive"
    COMBINED = "combined"


class Chronotype(str, Enum):
    MORNING = "morning"       # peak 7-11am, trough 2-5pm
    EVENING = "evening"       # peak 6-11pm, trough 7-10am
    NEUTRAL = "neutral"       # peak 10am-1pm, trough 3-4pm


class SeverityLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class CognitiveLoadCategory(str, Enum):
    LOW = "low"           # routine tasks, light reading, admin
    MEDIUM = "medium"     # moderate study, review, group work
    HIGH = "high"         # new material, problem sets, exams, writing


class TaskType(str, Enum):
    LECTURE = "lecture"
    STUDY = "study"
    EXAM = "exam"
    LAB = "lab"
    GROUP_WORK = "group_work"
    BREAK = "break"
    EXERCISE = "exercise"
    SOCIAL = "social"
    ADMIN = "admin"           # emails, planning, organizing
    CREATIVE = "creative"     # design, writing, brainstorming


class ScenarioType(str, Enum):
    LIGHT_WEEK = "light_week"
    HEAVY_WEEK = "heavy_week"
    EXAM_CRUNCH = "exam_crunch"
    POST_DEADLINE_RECOVERY = "post_deadline_recovery"
    CONFLICTING_PRIORITIES = "conflicting_priorities"
    FRAGMENTED_SCHEDULE = "fragmented_schedule"
    BACK_TO_BACK_CLASSES = "back_to_back_classes"
    MIDTERM_WEEK = "midterm_week"
    GROUP_PROJECT_DEADLINE = "group_project_deadline"
    PROCRASTINATION_RECOVERY = "procrastination_recovery"


# ============================================================================
# CORE MODELS
# ============================================================================

class TimeWindow(BaseModel):
    """A time window represented as hours (0-23)."""
    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=0, le=23)


class ADHDProfile(BaseModel):
    """
    ADHD student profile capturing cognitive and behavioral constraints
    relevant to schedule optimization.

    5 archetypes defined in ADHD_ARCHETYPES below.
    """
    profile_id: str
    name: str = Field(description="Human-readable archetype name")
    adhd_subtype: ADHDSubtype
    chronotype: Chronotype

    # Cognitive windows
    peak_hours: list[TimeWindow] = Field(
        description="Time windows when cognitive performance is highest"
    )
    trough_hours: list[TimeWindow] = Field(
        description="Time windows when cognitive performance is lowest (circadian trough)"
    )

    # Sustained attention
    max_sustained_focus_minutes: int = Field(
        ge=10, le=120,
        description="Max minutes of focused work before needing a break"
    )
    ideal_break_minutes: int = Field(
        ge=5, le=30,
        description="Optimal break duration to restore focus"
    )

    # Medication
    medication_managed: bool = False
    medication_active_window: Optional[TimeWindow] = Field(
        default=None,
        description="Hours when stimulant medication is active (if applicable)"
    )

    # Behavioral traits
    anxiety_level: SeverityLevel = SeverityLevel.LOW
    time_perception: SeverityLevel = Field(
        default=SeverityLevel.MODERATE,
        description="How poor time estimation is (HIGH = very poor)"
    )
    transition_difficulty: SeverityLevel = Field(
        default=SeverityLevel.MODERATE,
        description="Difficulty switching between dissimilar tasks"
    )
    hyperfocus_prone: bool = False
    procrastination_tendency: SeverityLevel = SeverityLevel.MODERATE

    # Scheduling preferences
    preferred_study_block_minutes: int = Field(
        default=45, ge=15, le=90,
        description="Preferred single study session length"
    )
    needs_buffer_between_tasks: bool = True
    min_buffer_minutes: int = Field(
        default=10, ge=5, le=30,
        description="Minimum transition buffer between dissimilar tasks"
    )


class ScheduleBlock(BaseModel):
    """A single block in a daily/weekly schedule."""
    title: str
    day: str = Field(description="Day of week: monday-sunday")
    start_time: str = Field(description="HH:MM in 24h format")
    end_time: str = Field(description="HH:MM in 24h format")
    cognitive_load: CognitiveLoadCategory
    task_type: TaskType
    course: Optional[str] = None
    is_fixed: bool = Field(
        default=False,
        description="True for immovable blocks (lectures, exams)"
    )
    is_decomposed: bool = Field(
        default=False,
        description="Whether this task has been broken into sub-tasks"
    )
    notes: Optional[str] = None


class Scenario(BaseModel):
    """
    A complete scheduling scenario: an ADHD profile + a week's schedule
    + context about what makes this week challenging.
    """
    scenario_id: str
    scenario_type: ScenarioType
    profile: ADHDProfile
    schedule: list[ScheduleBlock]
    week_context: str = Field(
        description="Narrative description of this week's situation"
    )
    expected_challenges: list[str] = Field(
        description="ADHD-specific challenges this scenario is designed to test"
    )
    optimal_interventions: list[str] = Field(
        default_factory=list,
        description="What an ideal ADHD-aware scheduler would do differently"
    )


# ============================================================================
# 5 ADHD PROFILE ARCHETYPES
# ============================================================================

ADHD_ARCHETYPES: list[ADHDProfile] = [
    ADHDProfile(
        profile_id="P1_inattentive_nightowl",
        name="Inattentive / Night Owl",
        adhd_subtype=ADHDSubtype.INATTENTIVE,
        chronotype=Chronotype.EVENING,
        peak_hours=[TimeWindow(start_hour=18, end_hour=23)],
        trough_hours=[TimeWindow(start_hour=7, end_hour=10)],
        max_sustained_focus_minutes=25,
        ideal_break_minutes=10,
        medication_managed=False,
        anxiety_level=SeverityLevel.LOW,
        time_perception=SeverityLevel.MODERATE,
        transition_difficulty=SeverityLevel.HIGH,
        hyperfocus_prone=True,
        procrastination_tendency=SeverityLevel.HIGH,
        preferred_study_block_minutes=30,
        needs_buffer_between_tasks=True,
        min_buffer_minutes=15,
    ),
    ADHDProfile(
        profile_id="P2_hyperactive_morning",
        name="Hyperactive / Morning Peak",
        adhd_subtype=ADHDSubtype.HYPERACTIVE,
        chronotype=Chronotype.MORNING,
        peak_hours=[TimeWindow(start_hour=7, end_hour=11)],
        trough_hours=[TimeWindow(start_hour=14, end_hour=17)],
        max_sustained_focus_minutes=20,
        ideal_break_minutes=5,
        medication_managed=False,
        anxiety_level=SeverityLevel.MODERATE,
        time_perception=SeverityLevel.LOW,
        transition_difficulty=SeverityLevel.LOW,
        hyperfocus_prone=False,
        procrastination_tendency=SeverityLevel.LOW,
        preferred_study_block_minutes=25,
        needs_buffer_between_tasks=True,
        min_buffer_minutes=10,
    ),
    ADHDProfile(
        profile_id="P3_combined_anxious",
        name="Combined / High Anxiety",
        adhd_subtype=ADHDSubtype.COMBINED,
        chronotype=Chronotype.NEUTRAL,
        peak_hours=[TimeWindow(start_hour=10, end_hour=12)],
        trough_hours=[
            TimeWindow(start_hour=14, end_hour=15),
            TimeWindow(start_hour=20, end_hour=22),
        ],
        max_sustained_focus_minutes=35,
        ideal_break_minutes=15,
        medication_managed=False,
        anxiety_level=SeverityLevel.HIGH,
        time_perception=SeverityLevel.MODERATE,
        transition_difficulty=SeverityLevel.MODERATE,
        hyperfocus_prone=True,
        procrastination_tendency=SeverityLevel.MODERATE,
        preferred_study_block_minutes=40,
        needs_buffer_between_tasks=True,
        min_buffer_minutes=15,
    ),
    ADHDProfile(
        profile_id="P4_inattentive_medicated",
        name="Inattentive / Medication-Managed",
        adhd_subtype=ADHDSubtype.INATTENTIVE,
        chronotype=Chronotype.MORNING,
        peak_hours=[
            TimeWindow(start_hour=9, end_hour=12),
            TimeWindow(start_hour=14, end_hour=16),
        ],
        trough_hours=[TimeWindow(start_hour=18, end_hour=21)],
        max_sustained_focus_minutes=50,
        ideal_break_minutes=10,
        medication_managed=True,
        medication_active_window=TimeWindow(start_hour=8, end_hour=16),
        anxiety_level=SeverityLevel.LOW,
        time_perception=SeverityLevel.LOW,
        transition_difficulty=SeverityLevel.LOW,
        hyperfocus_prone=False,
        procrastination_tendency=SeverityLevel.LOW,
        preferred_study_block_minutes=50,
        needs_buffer_between_tasks=True,
        min_buffer_minutes=10,
    ),
    ADHDProfile(
        profile_id="P5_combined_time_blind",
        name="Combined / Poor Time Perception",
        adhd_subtype=ADHDSubtype.COMBINED,
        chronotype=Chronotype.EVENING,
        peak_hours=[TimeWindow(start_hour=16, end_hour=20)],
        trough_hours=[
            TimeWindow(start_hour=8, end_hour=10),
            TimeWindow(start_hour=13, end_hour=14),
        ],
        max_sustained_focus_minutes=20,
        ideal_break_minutes=10,
        medication_managed=False,
        anxiety_level=SeverityLevel.MODERATE,
        time_perception=SeverityLevel.HIGH,
        transition_difficulty=SeverityLevel.HIGH,
        hyperfocus_prone=True,
        procrastination_tendency=SeverityLevel.HIGH,
        preferred_study_block_minutes=25,
        needs_buffer_between_tasks=True,
        min_buffer_minutes=15,
    ),
]


# ============================================================================
# 10 SCENARIO TYPES — descriptions for LLM generation
# ============================================================================

SCENARIO_TYPE_DESCRIPTIONS: dict[ScenarioType, str] = {
    ScenarioType.LIGHT_WEEK: (
        "A relaxed week with only 2-3 classes, no exams, minimal deadlines. "
        "Tests whether the scheduler avoids overloading and allows recovery time."
    ),
    ScenarioType.HEAVY_WEEK: (
        "An overloaded week: 5+ classes, 2 assignments due, possibly a quiz. "
        "Tests prioritization and cognitive load distribution."
    ),
    ScenarioType.EXAM_CRUNCH: (
        "2-3 exams within the same week. High cognitive load, high stakes. "
        "Tests spaced repetition placement and pre-exam buffer time."
    ),
    ScenarioType.POST_DEADLINE_RECOVERY: (
        "The week after a major deadline or exam period. Student is burned out. "
        "Tests whether scheduler reduces load and schedules lighter tasks."
    ),
    ScenarioType.CONFLICTING_PRIORITIES: (
        "Multiple competing obligations: group project meeting, part-time job shifts, "
        "assignment due, and a family event. Tests time-boxing and priority resolution."
    ),
    ScenarioType.FRAGMENTED_SCHEDULE: (
        "Classes scattered throughout the day with 30-60 min gaps between them. "
        "Tests whether the scheduler can use small gaps productively without "
        "overloading transitions."
    ),
    ScenarioType.BACK_TO_BACK_CLASSES: (
        "A day with 3-4 consecutive classes with no breaks. "
        "Tests transition buffer insertion and post-marathon recovery scheduling."
    ),
    ScenarioType.MIDTERM_WEEK: (
        "Multiple midterms spread across the week. Regular classes still happening. "
        "Tests study session interleaving and strategic rest placement."
    ),
    ScenarioType.GROUP_PROJECT_DEADLINE: (
        "A major group project due this week. Requires coordination meetings, "
        "individual work blocks, and integration sessions. Tests social task scheduling."
    ),
    ScenarioType.PROCRASTINATION_RECOVERY: (
        "Student has procrastinated on a large assignment due in 3 days. "
        "Must now cram while maintaining other commitments. "
        "Tests emergency rescheduling and task decomposition."
    ),
}
