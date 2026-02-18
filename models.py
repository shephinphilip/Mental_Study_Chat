#!/usr/bin/env python3
"""
Shared patient data models — imported by both api.py and crisis.py.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional


class RiskLevel(Enum):
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"
    CRISIS   = "crisis"


@dataclass
class ClinicalEssentials:
    """Patient profile for personalized psychiatric interview."""
    student_id: str
    name: str
    age: int
    current_class: str
    city: str
    chief_concern: str
    symptom_duration_weeks: int
    current_risk_level: RiskLevel
    suicidal_ideation: bool
    suicidal_intent: bool
    self_harm_history: bool
    self_harm_recent: bool
    substance_use_active: bool
    substance_of_choice: Optional[str]
    prior_diagnoses: List[str]
    current_medications: List[str]
    past_hospitalization: bool
    therapy_history: bool
    family_history_suicide: bool
    family_history_mental_illness: List[str]
    family_academic_pressure_level: str
    sibling_comparison_active: bool
    recent_marks_percentage: Optional[float]
    marks_trend: str
    subject_teacher_relationships: Dict[str, Dict]
    sleep_hours_last_week: int
    sleep_quality: str
    reasons_for_living: List[str]
    one_trusted_adult: Optional[str]
    recent_significant_events: List[str]
    ongoing_chronic_stressors: List[str]

    def get_age_bracket(self) -> str:
        if self.age <= 8:   return "child"
        elif self.age <= 11: return "latency"
        elif self.age <= 14: return "early_teen"
        else:                return "late_teen"

    def get_teacher_summary(self) -> str:
        parts = []
        for subj, data in self.subject_teacher_relationships.items():
            rel  = data.get("relationship", "neutral")
            name = data.get("name", "Unknown")
            if rel == "secure_base":      parts.append(f"{subj}({name}): supportive")
            elif rel == "anxiety_inducing": parts.append(f"{subj}({name}): fear-inducing")
            elif rel == "disengaged":       parts.append(f"{subj}({name}): unavailable")
            else:                           parts.append(f"{subj}({name}): neutral")
        return "; ".join(parts) if parts else "No data"

    def to_dict(self) -> dict:
        """Serialise to JSON-safe dict for API responses."""
        return {
            "student_id":   self.student_id,
            "name":         self.name,
            "age":          self.age,
            "current_class": self.current_class,
            "city":         self.city,
            "chief_concern": self.chief_concern,
            "current_risk_level": self.current_risk_level.value,
            "suicidal_ideation":  self.suicidal_ideation,
            "suicidal_intent":    self.suicidal_intent,
            "prior_diagnoses":    self.prior_diagnoses,
            "sleep_hours_last_week": self.sleep_hours_last_week,
            "recent_marks_percentage": self.recent_marks_percentage,
            "marks_trend":  self.marks_trend,
            "reasons_for_living": self.reasons_for_living,
            "one_trusted_adult":  self.one_trusted_adult,
            "ongoing_chronic_stressors": self.ongoing_chronic_stressors,
        }


# ─── Mock patient registry ────────────────────────────────────────────────────

MOCK_STUDENTS: Dict[str, ClinicalEssentials] = {

    "rohan_14": ClinicalEssentials(
        student_id="rohan_14", name="Rohan", age=14, current_class="10", city="Delhi",
        chief_concern="Board exam terror", symptom_duration_weeks=8,
        current_risk_level=RiskLevel.HIGH,
        suicidal_ideation=False, suicidal_intent=False,
        self_harm_history=False, self_harm_recent=False,
        substance_use_active=False, substance_of_choice=None,
        prior_diagnoses=["Test Anxiety"], current_medications=[],
        past_hospitalization=False, therapy_history=False,
        family_history_suicide=False, family_history_mental_illness=[],
        family_academic_pressure_level="extreme", sibling_comparison_active=True,
        recent_marks_percentage=76.0, marks_trend="Inconsistent",
        subject_teacher_relationships={
            "Mathematics":    {"name": "Mr. Aggarwal", "relationship": "secure_base",      "trauma": None,                        "impact": "positive"},
            "Science":        {"name": "Mrs. Khanna",  "relationship": "disengaged",        "trauma": "Teaches only top 5",        "impact": "negative"},
            "Social Science": {"name": "Mr. Banerjee", "relationship": "anxiety_inducing",  "trauma": "Not cut out for academics", "impact": "severe"},
        },
        sleep_hours_last_week=5, sleep_quality="Broken",
        reasons_for_living=["Cricket team", "Grandmother"],
        one_trusted_adult="Sports coach",
        recent_significant_events=["Pre-board results declined"],
        ongoing_chronic_stressors=["Coaching 6hrs daily"],
    ),

    "ananya_17": ClinicalEssentials(
        student_id="ananya_17", name="Ananya", age=17, current_class="Drop Year", city="Kota",
        chief_concern="I see no point in living if I don't get IIT", symptom_duration_weeks=24,
        current_risk_level=RiskLevel.CRISIS,
        suicidal_ideation=True, suicidal_intent=True,
        self_harm_history=True, self_harm_recent=True,
        substance_use_active=False, substance_of_choice=None,
        prior_diagnoses=["Major Depressive Disorder"], current_medications=[],
        past_hospitalization=False, therapy_history=True,
        family_history_suicide=False, family_history_mental_illness=["Father alcohol use"],
        family_academic_pressure_level="extreme", sibling_comparison_active=False,
        recent_marks_percentage=None, marks_trend="Crash",
        subject_teacher_relationships={
            "Physics":   {"name": "Mr. Suresh", "relationship": "anxiety_inducing", "trauma": "Public rank display, dropped A1→B3", "impact": "severe"},
            "Chemistry": {"name": "Mrs. Rekha",  "relationship": "disengaged",       "trauma": "Doesn't know my name",              "impact": "negative"},
        },
        sleep_hours_last_week=4, sleep_quality="Insomnia_onset",
        reasons_for_living=["Don't want to hurt parents"],
        one_trusted_adult=None,
        recent_significant_events=["Batch demotion 2 weeks ago"],
        ongoing_chronic_stressors=["Kota isolation", "Financial strain"],
    ),
}