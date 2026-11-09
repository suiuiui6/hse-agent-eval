from __future__ import annotations


SIDE_EFFECT_KEYS = frozenset(
    {
        "hse_report_task.insert",
        "hse_reminder.insert",
        "hse_weekly_report.insert",
        "hse_ai_alert.update",
    }
)

SIDE_EFFECT_ALIASES = {
    "task.insert": "hse_report_task.insert",
    "reminder.insert": "hse_reminder.insert",
    "weekly_report.insert": "hse_weekly_report.insert",
    "alert.read": "hse_ai_alert.update",
    "compliance.alert.insert": "hse_ai_alert.update",
}

KNOWN_SIDE_EFFECT_KEYS = SIDE_EFFECT_KEYS | frozenset(SIDE_EFFECT_ALIASES)


def canonical_side_effect_key(key: str) -> str:
    return SIDE_EFFECT_ALIASES.get(key, key)
