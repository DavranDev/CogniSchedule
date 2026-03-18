"""
Seed fake student schedule into the CogniSchedule backend database.
Reads research/data/fake_student_schedule.json and POSTs each block
to http://localhost:8000/api/events so they appear in the frontend calendar.

Usage:
    python research/data/seed_schedule.py
    python research/data/seed_schedule.py --clear   # delete all events first
"""

import argparse
import json
import sys
from pathlib import Path

import requests

API_BASE = "http://localhost:8000/api"

# Map task_type from fake schedule → event_type expected by backend
TASK_TYPE_MAP = {
    "lecture": "study",
    "study": "study",
    "exam": "study",
    "lab": "study",
    "group_work": "study",
    "break": "event",
    "exercise": "event",
    "social": "event",
    "admin": "event",
    "creative": "study",
}

# Map cognitive load → priority (optional enrichment)
LOAD_TO_PRIORITY = {
    "high": "high",
    "medium": "medium",
    "low": "low",
}

# Map task_type → calendar bucket
TASK_TO_CALENDAR = {
    "lecture": "study",
    "study": "study",
    "exam": "study",
    "lab": "study",
    "group_work": "study",
    "break": "personal",
    "exercise": "health",
    "social": "personal",
    "admin": "personal",
    "creative": "study",
}


def load_schedule(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def post_event(block: dict, date: str) -> dict:
    start_iso = f"{date}T{block['start_time']}:00"
    end_iso = f"{date}T{block['end_time']}:00"
    task_type = block.get("task_type", "event")

    payload = {
        "title": block["title"],
        "description": block.get("notes"),
        "start_time": start_iso,
        "end_time": end_iso,
        "calendar_id": TASK_TO_CALENDAR.get(task_type, "personal"),
        "all_day": False,
        "event_type": TASK_TYPE_MAP.get(task_type, "event"),
        "course": block.get("course"),
        "priority": LOAD_TO_PRIORITY.get(block.get("cognitive_load", "medium"), "medium"),
        "cognitive_load": block.get("cognitive_load"),
    }

    resp = requests.post(f"{API_BASE}/events", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def clear_all_events():
    events = requests.get(f"{API_BASE}/events", timeout=10).json()
    for ev in events:
        requests.delete(f"{API_BASE}/events/{ev['id']}", timeout=10)
    print(f"Cleared {len(events)} existing events.")


def main():
    parser = argparse.ArgumentParser(description="Seed fake schedule into CogniSchedule backend")
    parser.add_argument("--clear", action="store_true", help="Delete all existing events before seeding")
    args = parser.parse_args()

    schedule_path = Path(__file__).parent / "fake_student_schedule.json"
    if not schedule_path.exists():
        print(f"ERROR: {schedule_path} not found", file=sys.stderr)
        sys.exit(1)

    # Check backend is running
    try:
        requests.get(f"{API_BASE}/calendars", timeout=5)
    except requests.ConnectionError:
        print("ERROR: Backend not running at http://localhost:8000", file=sys.stderr)
        print("Start it with: cd calendar-app/backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000", file=sys.stderr)
        sys.exit(1)

    if args.clear:
        clear_all_events()

    data = load_schedule(schedule_path)
    student = data["student"]["name"]
    total = 0
    failed = 0

    for day_entry in data["schedule"]:
        date = day_entry["date"]
        day_name = day_entry["day"].capitalize()
        day_total = 0

        for block in day_entry["blocks"]:
            try:
                post_event(block, date)
                day_total += 1
                total += 1
            except Exception as e:
                print(f"  FAILED: {block['title']} — {e}", file=sys.stderr)
                failed += 1

        print(f"  {day_name} {date}: {day_total} events seeded")

    print(f"\nDone. Seeded {total} events for {student}. Failed: {failed}")
    if failed == 0:
        print("Refresh the calendar frontend to see them.")


if __name__ == "__main__":
    main()
