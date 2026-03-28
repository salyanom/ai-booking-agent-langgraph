"""LangGraph workflow for conversational booking."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent_tools import (
    BookingAgentState,
    book_meeting,
    cancel_meeting,
    check_availability,
    find_next_available_slots,
    list_booked_events,
    record_conflict_event,
    reschedule_meeting,
)


class ParsedIntent(BaseModel):
    action: Literal["book", "cancel", "reschedule", "unknown"] = Field(
        description="User intent. Use 'book', 'cancel', or 'reschedule' for clear scheduling intent."
    )
    start_time: str | None = Field(
        default=None,
        description="Start datetime in format YYYY-MM-DD HH:MM:SS.",
    )
    end_time: str | None = Field(
        default=None,
        description="End datetime in format YYYY-MM-DD HH:MM:SS.",
    )
    event_details: str = Field(
        default="Booking via AI assistant",
        description="Short event description from user message.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if the user request misses date/time information.",
    )
    clarification_question: str = Field(
        default="Could you share the exact date and time for the booking?",
        description="One follow-up question when clarification is needed.",
    )
    old_start_time: str | None = Field(default=None)
    old_end_time: str | None = Field(default=None)
    new_start_time: str | None = Field(default=None)
    new_end_time: str | None = Field(default=None)
    old_date: str | None = Field(default=None)


def _to_datetime_text(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _parse_date_from_text(user_text: str, now: datetime) -> datetime.date:
    lowered = user_text.lower()
    if "day after tomorrow" in lowered:
        return (now + timedelta(days=2)).date()
    if "tomorrow" in lowered:
        return (now + timedelta(days=1)).date()
    if "today" in lowered:
        return now.date()
    in_days_match = re.search(r"\bin\s+(\d{1,2})\s+days\b", lowered)
    if in_days_match:
        return (now + timedelta(days=int(in_days_match.group(1)))).date()
    if "next week" in lowered:
        return (now + timedelta(days=7)).date()

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for name, weekday in weekdays.items():
        if f"next {name}" in lowered or f"on {name}" in lowered or lowered.strip() == name or re.search(rf"\b{name}\b", lowered):
            delta = (weekday - now.weekday()) % 7
            if delta == 0:
                delta = 7
            return (now + timedelta(days=delta)).date()

    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", user_text)
    if date_match:
        return datetime.strptime(date_match.group(1), "%Y-%m-%d").date()

    slash_match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", user_text)
    if slash_match:
        token = slash_match.group(1)
        for fmt in ["%d/%m/%Y", "%m/%d/%Y"]:
            try:
                return datetime.strptime(token, fmt).date()
            except ValueError:
                continue
    dash_match = re.search(r"\b(\d{1,2}-\d{1,2}-\d{4})\b", user_text)
    if dash_match:
        token = dash_match.group(1)
        try:
            return datetime.strptime(token, "%d-%m-%Y").date()
        except ValueError:
            pass

    month_match = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
        lowered,
    )
    if month_match:
        month_name = month_match.group(1)
        day = int(month_match.group(2))
        year = int(month_match.group(3)) if month_match.group(3) else now.year
        month_map = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        try:
            result = datetime(year, month_map[month_name], day).date()
            if not month_match.group(3) and result < now.date():
                result = datetime(year + 1, month_map[month_name], day).date()
            return result
        except ValueError:
            pass

    return now.date()


def _parse_time_token(token: str, base_date: datetime.date) -> datetime | None:
    text = token.strip().lower().replace(".", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(\d{1,2})(am|pm)\b", r"\1 \2", text)

    named_times = {
        "noon": (12, 0),
        "midnight": (0, 0),
        "morning": (9, 0),
        "afternoon": (15, 0),
        "evening": (18, 0),
        "tonight": (20, 0),
    }
    if text in named_times:
        hour, minute = named_times[text]
        return datetime(
            year=base_date.year,
            month=base_date.month,
            day=base_date.day,
            hour=hour,
            minute=minute,
        )

    hour_only = re.fullmatch(r"([01]?\d|2[0-3])", text)
    if hour_only:
        return datetime(
            year=base_date.year,
            month=base_date.month,
            day=base_date.day,
            hour=int(hour_only.group(1)),
            minute=0,
        )

    formats = ["%I:%M %p", "%I %p", "%H:%M"]
    for fmt in formats:
        try:
            parsed = datetime.strptime(text.upper(), fmt)
            return datetime(
                year=base_date.year,
                month=base_date.month,
                day=base_date.day,
                hour=parsed.hour,
                minute=parsed.minute,
            )
        except ValueError:
            continue

    return None


def _extract_duration_minutes(user_text: str) -> int:
    lowered = user_text.lower()
    min_match = re.search(r"for\s+(\d{1,3})\s*(min|mins|minute|minutes)\b", lowered)
    if min_match:
        return int(min_match.group(1))

    hour_match = re.search(r"for\s+(\d{1,2})(?:\.(\d))?\s*(h|hr|hrs|hour|hours)\b", lowered)
    if hour_match:
        whole = int(hour_match.group(1))
        decimal = hour_match.group(2)
        if decimal:
            return int(float(f"{whole}.{decimal}") * 60)
        return whole * 60

    if "half hour" in lowered:
        return 30
    if "one hour" in lowered or "an hour" in lowered:
        return 60
    return 30


def _infer_action(user_text: str) -> str:
    lowered = user_text.lower()
    if any(k in lowered for k in ["cancel", "delete", "remove", "drop"]):
        return "cancel"
    if any(k in lowered for k in ["reschedule", "move", "shift", "push", "edit", "change", "update"]):
        return "reschedule"
    if any(k in lowered for k in ["book", "schedule", "set up", "arrange", "plan", "meeting", "sync", "call", "appointment"]):
        return "book"
    return "unknown"


def _extract_event_details(user_text: str) -> str:
    quoted_name_match = re.search(
        r"(?:meeting\s+name|name|title)\s*[\"']([^\"']+)[\"']",
        user_text,
        flags=re.IGNORECASE,
    )
    if quoted_name_match:
        details = quoted_name_match.group(1).strip()
        if details:
            return details

    named_match = re.search(
        r"(?:meeting\s+name|name|title)\s+([A-Za-z][A-Za-z\s\.-]{1,80})",
        user_text,
        flags=re.IGNORECASE,
    )
    if named_match:
        raw = named_match.group(1).strip()
        trimmed = re.split(r"\b(on|at|from|to|tomorrow|today|next)\b", raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if trimmed:
            return trimmed

    with_match = re.search(r"\bwith\s+([A-Za-z][A-Za-z\s\.-]{1,80})", user_text, flags=re.IGNORECASE)
    if with_match:
        raw = with_match.group(1).strip()
        trimmed = re.split(r"\b(on|at|from|to|tomorrow|today|next)\b", raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if trimmed:
            return trimmed

    detail_match = re.search(r"(?:for|about)\s+(.+)$", user_text, flags=re.IGNORECASE)
    if detail_match:
        details = detail_match.group(1).strip()
        if details:
            return details

    command_match = re.search(
        r"(?:cancel|book|schedule|reschedule|move|shift|push)\s+(.+)$",
        user_text,
        flags=re.IGNORECASE,
    )
    if command_match:
        details = command_match.group(1).strip()
        # Skip purely temporal phrases; keep only meaningful title-like fragments.
        if details and not re.search(r"\b(today|tomorrow|next|at|from|to|\d{1,2}:\d{2}|am|pm)\b", details.lower()):
            return details

    return "Booking via AI assistant"


def _parse_single_time_on_date(user_text: str, date_value: datetime.date) -> tuple[str, str] | None:
    lowered = user_text.lower()
    token_match = re.search(
        r"(?:at\s+)?(noon|midnight|morning|afternoon|evening|tonight|[0-9]{1,2}:[0-9]{2}\s*(?:am|pm)?|[0-9]{1,2}\s*(?:am|pm))",
        lowered,
    )
    if not token_match:
        return None

    start_dt = _parse_time_token(token_match.group(1), date_value)
    if not start_dt:
        return None

    duration = _extract_duration_minutes(user_text)
    end_dt = start_dt + timedelta(minutes=duration)
    return _to_datetime_text(start_dt), _to_datetime_text(end_dt)


def _find_event_by_details_fragment(fragment: str) -> dict | None:
    frag = fragment.strip().lower()
    if not frag:
        return None

    # Drop common words so title matching is less brittle.
    cleaned = re.sub(r"\b(reschedule|move|shift|edit|change|update|booking|meeting|my|the|please|to)\b", " ", frag)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None

    for item in list_booked_events():
        details = str(item.get("details", "")).lower()
        if cleaned in details:
            return item
    return None


def _is_smalltalk_or_capability_request(user_text: str) -> bool:
    lowered = user_text.lower().strip()
    if lowered in {"hi", "hey", "hello", "yo", "sup", "hola"}:
        return True

    cues = [
        "help",
        "what can you do",
        "how can you help",
        "capabilities",
        "who are you",
        "thanks",
        "thank you",
    ]
    return any(cue in lowered for cue in cues)


def _heuristic_parse_intent(user_text: str, now: datetime) -> dict | None:
    action = _infer_action(user_text)
    if action == "unknown":
        return None

    fragment = _parse_time_window_fragment(user_text, now)
    if not fragment:
        return {
            "action": action,
            "event_details": _extract_event_details(user_text),
            "needs_clarification": True,
            "clarification_question": "I can help with that. What date and time should I use?",
        }

    start_time, end_time = fragment
    details = _extract_event_details(user_text)

    if action == "reschedule":
        return {
            "action": "reschedule",
            "start_time": start_time,
            "end_time": end_time,
            "new_start_time": start_time,
            "new_end_time": end_time,
            "event_details": details,
            "needs_clarification": True,
            "clarification_question": "Please share the original meeting time to move.",
        }

    return {
        "action": action,
        "start_time": start_time,
        "end_time": end_time,
        "event_details": details,
        "needs_clarification": False,
        "clarification_question": "",
    }


def _regex_parse_intent(user_text: str, now: datetime) -> dict | None:
    lowered = user_text.lower()
    cancel_keywords = ["cancel", "delete", "remove", "drop"]
    is_cancel = any(keyword in lowered for keyword in cancel_keywords)
    reschedule_keywords = ["reschedule", "move", "shift", "push", "edit", "change", "update"]
    is_reschedule = any(keyword in lowered for keyword in reschedule_keywords)

    action_keywords = ["book", "schedule", "set up", "arrange", "plan", "meeting", "sync", "call", "appointment"]
    is_booking = any(keyword in lowered for keyword in action_keywords)
    if not (is_cancel or is_booking or is_reschedule):
        return None

    base_date = _parse_date_from_text(user_text, now)

    range_match = re.search(
        r"from\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)\s+to\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)",
        lowered,
    )

    if is_reschedule and " to " in lowered:
        old_part, new_part = lowered.split(" to ", 1)
        old_fragment = _parse_time_window_fragment(old_part, now)
        new_fragment = _parse_time_window_fragment(new_part, now)
        if old_fragment and new_fragment:
            details_match = re.search(r"(?:reschedule|move|shift)\s+(.+?)\s+to", user_text, flags=re.IGNORECASE)
            details = details_match.group(1).strip() if details_match else "Booking via AI assistant"
            return {
                "action": "reschedule",
                "old_start_time": old_fragment[0],
                "old_end_time": old_fragment[1],
                "new_start_time": new_fragment[0],
                "new_end_time": new_fragment[1],
                "start_time": new_fragment[0],
                "end_time": new_fragment[1],
                "event_details": details,
                "needs_clarification": False,
                "clarification_question": "",
            }

        # Handle "reschedule <meeting name> to <new time>".
        if not old_fragment and new_fragment:
            matched = _find_event_by_details_fragment(old_part)
            if matched:
                old_start = str(matched["start_time"])
                old_end = str(matched["end_time"])
                old_start_dt = datetime.strptime(old_start, "%Y-%m-%d %H:%M:%S")
                old_end_dt = datetime.strptime(old_end, "%Y-%m-%d %H:%M:%S")
                new_start_dt = datetime.strptime(new_fragment[0], "%Y-%m-%d %H:%M:%S")
                duration_minutes = max(30, int((old_end_dt - old_start_dt).total_seconds() // 60))
                new_end_dt = new_start_dt + timedelta(minutes=duration_minutes)
                return {
                    "action": "reschedule",
                    "old_start_time": old_start,
                    "old_end_time": old_end,
                    "new_start_time": _to_datetime_text(new_start_dt),
                    "new_end_time": _to_datetime_text(new_end_dt),
                    "start_time": _to_datetime_text(new_start_dt),
                    "end_time": _to_datetime_text(new_end_dt),
                    "event_details": str(matched.get("details") or "Booking via AI assistant"),
                    "needs_clarification": False,
                    "clarification_question": "",
                }

    if is_reschedule:
        shift_match = re.search(r"by\s+(\d{1,3})\s*(min|mins|minute|minutes|h|hr|hrs|hour|hours)", lowered)
        base_fragment = _parse_time_window_fragment(user_text, now)
        if shift_match and base_fragment:
            unit = shift_match.group(2)
            amount = int(shift_match.group(1))
            delta_minutes = amount if unit.startswith("min") else amount * 60

            old_start = datetime.strptime(base_fragment[0], "%Y-%m-%d %H:%M:%S")
            old_end = datetime.strptime(base_fragment[1], "%Y-%m-%d %H:%M:%S")
            new_start = old_start + timedelta(minutes=delta_minutes)
            new_end = old_end + timedelta(minutes=delta_minutes)
            return {
                "action": "reschedule",
                "old_start_time": base_fragment[0],
                "old_end_time": base_fragment[1],
                "new_start_time": _to_datetime_text(new_start),
                "new_end_time": _to_datetime_text(new_end),
                "start_time": _to_datetime_text(new_start),
                "end_time": _to_datetime_text(new_end),
                "event_details": "Booking via AI assistant",
                "needs_clarification": False,
                "clarification_question": "",
            }

        has_time_hint = bool(
            re.search(
                r"\b([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)|noon|midnight|morning|afternoon|evening|tonight)\b",
                lowered,
            )
        )
        if ("reschedule" in lowered or "edit" in lowered or "change" in lowered or "update" in lowered) and not has_time_hint:
            details = _extract_event_details(user_text)
            return {
                "action": "reschedule",
                "event_details": details,
                "old_date": base_date.strftime("%Y-%m-%d"),
                "needs_clarification": True,
                "clarification_question": "Please share the original start time for that date.",
            }

    if range_match:
        start_dt = _parse_time_token(range_match.group(1), base_date)
        end_dt = _parse_time_token(range_match.group(2), base_date)
        if start_dt and end_dt and end_dt > start_dt:
            details = _extract_event_details(user_text)
            return {
                "action": "cancel" if is_cancel else "book",
                "start_time": _to_datetime_text(start_dt),
                "end_time": _to_datetime_text(end_dt),
                "event_details": details,
                "needs_clarification": False,
                "clarification_question": "",
            }

    at_match = re.search(
        r"(?:at\s+)?(noon|midnight|morning|afternoon|evening|tonight|[0-9]{1,2}:[0-9]{2}\s*(?:am|pm)?|[0-9]{1,2}\s*(?:am|pm))",
        lowered,
    )
    if at_match:
        start_dt = _parse_time_token(at_match.group(1), base_date)
        if start_dt:
            duration = _extract_duration_minutes(user_text)
            end_dt = start_dt + timedelta(minutes=duration)
            details = _extract_event_details(user_text)
            return {
                "action": "cancel" if is_cancel else "book",
                "start_time": _to_datetime_text(start_dt),
                "end_time": _to_datetime_text(end_dt),
                "event_details": details,
                "needs_clarification": False,
                "clarification_question": "",
            }

    return None


def _parse_conflict_selection(
    user_text: str,
    conflict_suggestions: list[str],
) -> tuple[str, str] | None:
    if not conflict_suggestions:
        return None

    lowered = user_text.lower().strip()
    idx: int | None = None

    option_match = re.search(r"\b(option\s*)?(\d+)\b", lowered)
    if option_match:
        idx = int(option_match.group(2)) - 1

    if idx is None:
        if "first" in lowered or "1st" in lowered:
            idx = 0
        elif "second" in lowered or "2nd" in lowered:
            idx = 1
        elif "third" in lowered or "3rd" in lowered:
            idx = 2
        elif lowered in {"yes", "ok", "okay", "sure", "book it", "confirm"}:
            idx = 0

    if idx is None:
        for i, suggestion in enumerate(conflict_suggestions):
            start_token = suggestion.split(" to ")[0]
            if start_token in user_text:
                idx = i
                break

    if idx is None:
        return None

    if idx < 0 or idx >= len(conflict_suggestions):
        return None

    selected = conflict_suggestions[idx]
    split_match = re.match(
        r"\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+to\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*",
        selected,
    )
    if not split_match:
        return None

    return split_match.group(1), split_match.group(2)

def _parse_cancel_selection(
    user_text: str,
    suggestions: list[dict],
) -> dict | None:
    if not suggestions:
        return None

    lowered = user_text.lower().strip()

    # Option number support
    option_match = re.search(r"\b(option\s*)?(\d+)\b", lowered)
    if option_match:
        idx = int(option_match.group(2)) - 1
        if 0 <= idx < len(suggestions):
            return suggestions[idx]

    # Text-based match
    for item in suggestions:
        start_time = str(item.get("start_time", ""))
        end_time = str(item.get("end_time", ""))
        details = str(item.get("details", "")).lower()

        if start_time in user_text:
            return item

        if end_time in user_text:
            return item

        if details and details in lowered:
            return item

    # Relative choices
    if lowered in {"this", "cancel this", "the last one", "last one"}:
        return suggestions[0]

    return None


def _parse_reschedule_selection(
    user_text: str,
    suggestions: list[dict],
) -> dict | None:
    return _parse_cancel_selection(user_text, suggestions)

def _get_last_human_message(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _get_previous_human_message(messages: list[BaseMessage]) -> str:
    seen = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            seen += 1
            if seen == 2:
                return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _get_last_ai_message(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _parse_time_window_fragment(
    user_text: str,
    now: datetime,
) -> tuple[str, str] | None:
    base_date = _parse_date_from_text(user_text, now)
    lowered = user_text.lower()

    range_match = re.search(
        r"from\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)\s+to\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)",
        lowered,
    )
    if range_match:
        start_dt = _parse_time_token(range_match.group(1), base_date)
        end_dt = _parse_time_token(range_match.group(2), base_date)
        if start_dt and end_dt and end_dt > start_dt:
            return _to_datetime_text(start_dt), _to_datetime_text(end_dt)

    between_match = re.search(
        r"between\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)\s+and\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)",
        lowered,
    )
    if between_match:
        start_dt = _parse_time_token(between_match.group(1), base_date)
        end_dt = _parse_time_token(between_match.group(2), base_date)
        if start_dt and end_dt and end_dt > start_dt:
            return _to_datetime_text(start_dt), _to_datetime_text(end_dt)

    at_match = re.search(
        r"(?:at\s+)?(noon|midnight|morning|afternoon|evening|tonight|[0-9]{1,2}:[0-9]{2}\s*(?:am|pm)?|[0-9]{1,2}\s*(?:am|pm))",
        lowered,
    )
    if at_match:
        start_dt = _parse_time_token(at_match.group(1), base_date)
        if start_dt:
            duration = _extract_duration_minutes(user_text)
            end_dt = start_dt + timedelta(minutes=duration)
            return _to_datetime_text(start_dt), _to_datetime_text(end_dt)

    return None


def _build_clarification_message(intent: dict, last_ai: str) -> str:
    start_time = intent.get("start_time")
    end_time = intent.get("end_time")
    action = intent.get("action", "book")

    if not start_time and not end_time:
        msg = (
            "Please share a time window. Example: 'tomorrow at 3 PM for 30 min' "
            "or '2026-04-02 from 14:00 to 14:30'."
        )
    elif start_time and not end_time:
        msg = (
            "I have the start time. Please provide an end time or duration. "
            "Example: 'for 45 min' or 'to 4 PM'."
        )
    else:
        msg = (
            "Please confirm details for this request, including date and time."
        )

    if action == "cancel":
        msg = msg.replace("Please share a time window", "Please share the meeting time to cancel")

    if last_ai and msg.strip() == last_ai.strip():
        return f"{msg} If easier, reply with: tomorrow 3 PM for 30 min."
    return msg


def _build_general_assistant_reply(llm: BaseChatModel, user_text: str) -> str:
    _ = llm
    _ = user_text
    return (
        "I can help you book, cancel, and reschedule meetings in natural language. "
        "Examples: 'book next Friday afternoon', 'cancel design sync', or 'move tomorrow 3pm call to 5pm'."
    )


def _create_llm(
    model_name: str | None = None,
    provider: str | None = None,
) -> BaseChatModel:
    selected_provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()

    if selected_provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "Ollama provider selected but langchain-ollama is not installed."
            ) from exc

        selected_model = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
        return ChatOllama(model=selected_model, temperature=0)

    selected_model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=selected_model, temperature=0)


def intent_parser_node(state: BookingAgentState, llm: BaseChatModel) -> dict:
    messages = state.get("messages", [])
    user_text = _get_last_human_message(messages)

    if not user_text:
        return {
            "booking_status": "needs_clarification",
            "messages": [
                AIMessage(content="Please tell me what date and time you want to book.")
            ],
        }

    prior_intent = state.get("current_intent", {})

    # Allow users to explicitly proceed with a conflicting slot and store it as conflict.
    force_conflict_phrases = [
        "go with conflict",
        "go ahead with conflict",
        "book anyway",
        "proceed anyway",
        "force booking",
        "keep conflict",
    ]
    if (
        state.get("booking_status") == "conflict"
        and str(prior_intent.get("action", "book")) in {"book", "reschedule"}
        and any(phrase in user_text.lower() for phrase in force_conflict_phrases)
    ):
        return {
            "current_intent": {
                **prior_intent,
                "allow_conflict": True,
                "needs_clarification": False,
            },
            "booking_status": "pending",
        }

    if _is_smalltalk_or_capability_request(user_text) and not prior_intent.get("action"):
        return {
            "current_intent": {"action": "unknown"},
            "booking_status": "needs_clarification",
            "conflict_suggestions": [],
            "messages": [AIMessage(content=_build_general_assistant_reply(llm, user_text))],
        }

    # If user says "move it to ..." after an existing booking intent, reuse prior window.
    if prior_intent.get("start_time") and prior_intent.get("end_time"):
        lowered = user_text.lower()
        if any(k in lowered for k in ["move it", "move this", "shift it", "reschedule it", "push it"]):
            new_fragment = _parse_time_window_fragment(user_text, datetime.now())
            if new_fragment:
                return {
                    "current_intent": {
                        **prior_intent,
                        "action": "reschedule",
                        "old_start_time": prior_intent.get("start_time"),
                        "old_end_time": prior_intent.get("end_time"),
                        "new_start_time": new_fragment[0],
                        "new_end_time": new_fragment[1],
                        "start_time": new_fragment[0],
                        "end_time": new_fragment[1],
                        "needs_clarification": False,
                    },
                    "booking_status": "pending",
                }
    inferred_now = _infer_action(user_text)

    generic_edit_request = bool(
        re.search(r"\b(edit|change|update|reschedule)\b", user_text.lower())
        and re.search(r"\b(booking|meeting)\b", user_text.lower())
    )
    has_any_time_or_date = bool(
        re.search(
            r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-\d{1,2}-\d{4}|[0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)|noon|midnight|morning|afternoon|evening|tonight|today|tomorrow|day after tomorrow|next\s+\w+)\b",
            user_text.lower(),
        )
    )
    if inferred_now == "reschedule" and generic_edit_request and not has_any_time_or_date:
        return {
            "current_intent": {
                "action": "reschedule",
                "event_details": "Booking via AI assistant",
                "needs_clarification": False,
            },
            "booking_status": "pending",
            "conflict_suggestions": [],
        }

    if (
        state.get("booking_status") == "needs_clarification"
        and prior_intent.get("action") in {"book", "cancel", "reschedule"}
        and not (
            prior_intent.get("action") in {"cancel", "reschedule"}
            and inferred_now in {"book", "cancel", "reschedule"}
            and inferred_now != prior_intent.get("action")
        )
    ):
        merged_intent = dict(prior_intent)

        # When user is choosing from listed reschedule options, prioritize explicit
        # option/title selection before interpreting the message as a new time fragment.
        if merged_intent.get("action") == "reschedule" and not merged_intent.get("old_start_time"):
            picked_reschedule = _parse_reschedule_selection(
                user_text,
                state.get("reschedule_suggestions", []),
            )
            if picked_reschedule:
                merged_intent["old_start_time"] = picked_reschedule["start_time"]
                merged_intent["old_end_time"] = picked_reschedule["end_time"]
                merged_intent["event_details"] = picked_reschedule["details"]
                merged_intent["needs_clarification"] = True
                return {
                    "current_intent": merged_intent,
                    "booking_status": "needs_clarification",
                    "messages": [
                        AIMessage(content="Got it. What is the new date/time you want to move it to?")
                    ],
                }

        if merged_intent.get("action") == "reschedule":
            old_date_text = str(merged_intent.get("old_date") or "").strip()
            if old_date_text and not merged_intent.get("old_start_time"):
                try:
                    old_date = datetime.strptime(old_date_text, "%Y-%m-%d").date()
                    old_fragment = _parse_single_time_on_date(user_text, old_date)
                    if old_fragment:
                        merged_intent["old_start_time"], merged_intent["old_end_time"] = old_fragment
                        return {
                            "current_intent": merged_intent,
                            "booking_status": "needs_clarification",
                            "messages": [
                                AIMessage(
                                    content="Got it. What is the new date/time you want to move it to?"
                                )
                            ],
                        }
                except ValueError:
                    pass

            if merged_intent.get("old_start_time") and not merged_intent.get("new_start_time"):
                old_start_text = str(merged_intent.get("old_start_time") or "")
                explicit_date_in_reply = bool(
                    re.search(
                        r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-\d{1,2}-\d{4}|today|tomorrow|day after tomorrow|next\s+\w+)\b",
                        user_text.lower(),
                    )
                )
                if old_start_text and not explicit_date_in_reply:
                    try:
                        old_date = datetime.strptime(old_start_text, "%Y-%m-%d %H:%M:%S").date()
                        same_day_fragment = _parse_single_time_on_date(user_text, old_date)
                        if same_day_fragment:
                            merged_intent["new_start_time"], merged_intent["new_end_time"] = same_day_fragment
                            merged_intent["start_time"], merged_intent["end_time"] = same_day_fragment
                            merged_intent["needs_clarification"] = False
                            return {
                                "current_intent": merged_intent,
                                "booking_status": "pending",
                            }
                    except ValueError:
                        pass

                new_fragment = _parse_time_window_fragment(user_text, datetime.now())
                if new_fragment:
                    merged_intent["new_start_time"], merged_intent["new_end_time"] = new_fragment
                    merged_intent["start_time"], merged_intent["end_time"] = new_fragment
                    merged_intent["needs_clarification"] = False
                    return {
                        "current_intent": merged_intent,
                        "booking_status": "pending",
                    }

        fragment = _parse_time_window_fragment(user_text, datetime.now())
        if fragment:
            merged_intent["start_time"], merged_intent["end_time"] = fragment
            merged_intent["needs_clarification"] = False
            return {
                "current_intent": merged_intent,
                "booking_status": "pending",
            }

        start_time = merged_intent.get("start_time")
        if start_time:
            start_dt = datetime.strptime(str(start_time), "%Y-%m-%d %H:%M:%S")
            to_match = re.search(r"\bto\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm))", user_text.lower())
            if to_match:
                end_dt = _parse_time_token(to_match.group(1), start_dt.date())
                if end_dt and end_dt > start_dt:
                    merged_intent["end_time"] = _to_datetime_text(end_dt)
                    merged_intent["needs_clarification"] = False
                    return {
                        "current_intent": merged_intent,
                        "booking_status": "pending",
                    }

            duration_match = re.search(r"for\s+(.+)$", user_text.lower())
            if duration_match:
                duration_minutes = _extract_duration_minutes(user_text)
                if duration_minutes > 0:
                    end_dt = start_dt + timedelta(minutes=duration_minutes)
                    merged_intent["end_time"] = _to_datetime_text(end_dt)
                    merged_intent["needs_clarification"] = False
                    return {
                        "current_intent": merged_intent,
                        "booking_status": "pending",
                    }
                
    lowered_user = user_text.lower().strip()
    is_selection_reply = bool(re.search(r"\b(option\s*)?\d+\b", lowered_user)) or lowered_user in {
        "this",
        "cancel this",
        "the last one",
        "last one",
    }

    cancel_selection = None
    if state.get("booking_status") == "needs_clarification" and str(prior_intent.get("action", "")) == "cancel":
        cancel_selection = _parse_cancel_selection(
            user_text,
            state.get("cancel_suggestions", []),
        )

        if not cancel_selection and is_selection_reply:
            fallback_options = list_booked_events()[:5]
            cancel_selection = _parse_cancel_selection(user_text, fallback_options)

    if cancel_selection:
        return {
            "current_intent": {
                "action": "cancel",
                "start_time": cancel_selection["start_time"],
                "end_time": cancel_selection["end_time"],
                "event_details": cancel_selection["details"],
                "needs_clarification": False,
            },
            "conflict_suggestions": [],
            "booking_status": "pending",
        }

    reschedule_selection = None
    if state.get("booking_status") == "needs_clarification" and str(prior_intent.get("action", "")) == "reschedule":
        reschedule_selection = _parse_reschedule_selection(
            user_text,
            state.get("reschedule_suggestions", []),
        )

        if not reschedule_selection and is_selection_reply:
            fallback_options = list_booked_events()[:5]
            reschedule_selection = _parse_reschedule_selection(user_text, fallback_options)

    if reschedule_selection:
        return {
            "current_intent": {
                **prior_intent,
                "action": "reschedule",
                "old_start_time": reschedule_selection["start_time"],
                "old_end_time": reschedule_selection["end_time"],
                "event_details": reschedule_selection["details"],
                "needs_clarification": True,
            },
            "conflict_suggestions": [],
            "reschedule_suggestions": state.get("reschedule_suggestions", []),
            "booking_status": "needs_clarification",
            "messages": [
                AIMessage(content="Got it. What is the new date/time you want to move it to?")
            ],
        }

    selected_option = None
    if str(prior_intent.get("action", "")) != "cancel":
        selected_option = _parse_conflict_selection(
            user_text,
            state.get("conflict_suggestions", []),
        )
    if selected_option:
        start_time, end_time = selected_option
        prior_intent = state.get("current_intent", {})
        prior_action = str(prior_intent.get("action", "book"))
        details = str(prior_intent.get("event_details") or "Booking via AI assistant")
        if details == "Booking via AI assistant":
            previous_human = _get_previous_human_message(messages)
            extracted = _extract_event_details(previous_human)
            if extracted and extracted != "Booking via AI assistant":
                details = extracted
        if prior_action == "reschedule":
            return {
                "current_intent": {
                    **prior_intent,
                    "action": "reschedule",
                    "new_start_time": start_time,
                    "new_end_time": end_time,
                    "start_time": start_time,
                    "end_time": end_time,
                    "event_details": details,
                    "needs_clarification": False,
                },
                "booking_status": "pending",
            }

        return {
            "current_intent": {
                **prior_intent,
                "action": prior_action if prior_action in {"book", "cancel"} else "book",
                "start_time": start_time,
                "end_time": end_time,
                "event_details": details,
                "needs_clarification": False,
            },
            "booking_status": "pending",
        }

    now = datetime.now()
    regex_intent = _regex_parse_intent(user_text, now)
    if regex_intent:
        if regex_intent.get("needs_clarification"):
            question = str(regex_intent.get("clarification_question") or "Please share more scheduling details.")
            return {
                "current_intent": regex_intent,
                "booking_status": "needs_clarification",
                "messages": [AIMessage(content=question)],
            }
        return {
            "current_intent": regex_intent,
            "booking_status": "pending",
        }

    # Model-independent fallback: allow cancellation by meeting title without exact time.
    inferred = _infer_action(user_text)
    extracted_details = _extract_event_details(user_text)
    if inferred == "cancel" and extracted_details != "Booking via AI assistant":
        return {
            "current_intent": {
                "action": "cancel",
                "event_details": extracted_details,
                "needs_clarification": False,
            },
            "booking_status": "pending",
        }

    now_text = now.strftime("%Y-%m-%d %H:%M:%S")

    try:
        parser = llm.with_structured_output(ParsedIntent)
        parsed = parser.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an intent parser for a booking agent. "
                        "Extract schedule intent and normalize all times using this format: "
                        "YYYY-MM-DD HH:MM:SS. "
                        f"Current local datetime: {now_text}. "
                        "If the request is vague, set needs_clarification=true and ask one concise, specific follow-up question. "
                        "When a user gives start time but no end, default to a 30-minute meeting."
                    )
                ),
                HumanMessage(content=user_text),
            ]
        )
    except Exception:
        heuristic = _heuristic_parse_intent(user_text, now)
        if heuristic:
            return {
                "current_intent": heuristic,
                "booking_status": "pending" if not heuristic.get("needs_clarification") else "needs_clarification",
                "messages": [AIMessage(content=heuristic.get("clarification_question", ""))]
                if heuristic.get("needs_clarification")
                else [],
            }

        general_reply = _build_general_assistant_reply(llm, user_text)
        return {
            "current_intent": {"action": "unknown"},
            "booking_status": "needs_clarification",
            "conflict_suggestions": [],
            "messages": [AIMessage(content=general_reply)],
        }

    if parsed.action == "book" and parsed.start_time and not parsed.end_time:
        start_dt = datetime.strptime(parsed.start_time, "%Y-%m-%d %H:%M:%S")
        parsed.end_time = _to_datetime_text(start_dt + timedelta(minutes=30))
        parsed.needs_clarification = False

    if parsed.action == "cancel" and parsed.start_time and not parsed.end_time:
        start_dt = datetime.strptime(parsed.start_time, "%Y-%m-%d %H:%M:%S")
        parsed.end_time = _to_datetime_text(start_dt + timedelta(minutes=30))
        parsed.needs_clarification = False

    if (
        parsed.action not in {"book", "cancel", "reschedule"}
        or parsed.needs_clarification
        or not parsed.start_time
        or not parsed.end_time
    ):
        last_ai = _get_last_ai_message(messages)
        merged = parsed.model_dump()
        if prior_intent.get("action") in {"book", "cancel", "reschedule"} and merged.get("action") == "unknown":
            merged["action"] = prior_intent.get("action")

        # If user is not in an active booking/cancel clarification flow,
        # reply conversationally instead of repeatedly forcing date/time prompts.
        if merged.get("action") == "unknown" and prior_intent.get("action") not in {"book", "cancel", "reschedule"}:
            general_reply = _build_general_assistant_reply(llm, user_text)
            return {
                "current_intent": merged,
                "booking_status": "needs_clarification",
                "conflict_suggestions": [],
                "messages": [AIMessage(content=general_reply)],
            }

        if _is_smalltalk_or_capability_request(user_text) and merged.get("action") in {"unknown", "book", "cancel", "reschedule"}:
            return {
                "current_intent": {"action": "unknown"},
                "booking_status": "needs_clarification",
                "conflict_suggestions": [],
                "messages": [AIMessage(content=_build_general_assistant_reply(llm, user_text))],
            }

        clarification_text = (
            parsed.clarification_question
            if parsed.clarification_question and parsed.clarification_question.strip()
            else _build_clarification_message(merged, last_ai)
        )

        return {
            "current_intent": merged,
            "booking_status": "needs_clarification",
            "conflict_suggestions": [],
            "messages": [
                AIMessage(
                    content=clarification_text
                )
            ],
        }

    return {
        "current_intent": parsed.model_dump(),
        "booking_status": "pending",
        "conflict_suggestions": [],
    }


def cancellation_confirmer_node(state: BookingAgentState) -> dict:
    intent = state.get("current_intent", {})
    start_time = intent.get("start_time")
    end_time = intent.get("end_time")
    details = intent.get("event_details")
    latest_user = _get_last_human_message(state.get("messages", []))
    suggested = _parse_cancel_selection(latest_user, state.get("cancel_suggestions", []))

    if suggested:
        start_time = suggested.get("start_time")
        end_time = suggested.get("end_time")
        details = suggested.get("details")

    if not start_time or not end_time:
        upcoming = list_booked_events()[:5]
        details_text = str(details or "").strip().lower()

        if details_text in {"", "booking via ai assistant", "booking", "meeting"}:
            details_text = ""

        if details_text and details_text != "booking via ai assistant":
            matches = [
                item for item in upcoming if details_text in str(item.get("details", "")).lower()
            ]
            if len(matches) == 1:
                picked = matches[0]
                result = cancel_meeting(
                    str(picked["start_time"]),
                    str(picked["end_time"]),
                    str(picked["details"]),
                )
                if result.get("success"):
                    summary = result["summary"]
                    return {
                        "booking_status": "cancelled",
                        "cancel_suggestions": [],
                        "messages": [
                            AIMessage(
                                content=(
                                    "Cancellation confirmed.\n"
                                    f"- Start: {summary['start_time']}\n"
                                    f"- End: {summary['end_time']}\n"
                                    f"- Slots released: {summary['cancelled_slots']}"
                                )
                            )
                        ],
                    }

            if len(matches) > 1:
                options = "\n".join(
                    f"{i}. {item['start_time']} to {item['end_time']} ({item['details']})"
                    for i, item in enumerate(matches, start=1)
                )
                return {
                    "booking_status": "needs_clarification",
                    "cancel_suggestions": matches,
                    "conflict_suggestions": [],
                    "messages": [
                        AIMessage(
                            content=(
                                "I found multiple matching bookings. Choose one to cancel:\n"
                                f"{options}\n\n"
                                "Reply with option number."
                            )
                        )
                    ],
                }

        if upcoming:
            options = "\n".join(
                f"{i}. {item['start_time']} to {item['end_time']} ({item['details']})"
                for i, item in enumerate(upcoming, start=1)
            )

            return {
                "booking_status": "needs_clarification",
                "cancel_suggestions": upcoming,
                "conflict_suggestions": [],
                "messages": [
                    AIMessage(
                        content=(
                            "Please choose which booking to cancel:\n"
                            f"{options}\n\n"
                            "Reply with option number or booking name."
                        )
                    )
                ],
            }

        return {
            "booking_status": "needs_clarification",
            "messages": [
                AIMessage(
                    content="Please share the meeting time window you want to cancel."
                )
            ],
        }

    result = cancel_meeting(start_time, end_time, details)

    if not result["success"]:
        if suggested:
            retry = cancel_meeting(str(start_time), str(end_time), None)
            if retry.get("success"):
                summary = retry["summary"]
                return {
                    "booking_status": "cancelled",
                    "cancel_suggestions": [],
                    "conflict_suggestions": [],
                    "messages": [
                        AIMessage(
                            content=(
                                "Cancellation confirmed.\n"
                                f"- Start: {summary['start_time']}\n"
                                f"- End: {summary['end_time']}\n"
                                f"- Slots released: {summary['cancelled_slots']}"
                            )
                        )
                    ],
                }

        upcoming = list_booked_events()[:5]

        if upcoming:
            options = "\n".join(
                f"{i}. {item['start_time']} to {item['end_time']} ({item['details']})"
                for i, item in enumerate(upcoming, start=1)
            )

            return {
                "booking_status": "needs_clarification",
                "cancel_suggestions": upcoming,
                "conflict_suggestions": [],
                "messages": [
                    AIMessage(
                        content=(
                            f"{result['message']}\n\n"
                            "Try one of these bookings:\n"
                            f"{options}\n\n"
                            "Reply with option number."
                        )
                    )
                ],
            }

        return {
            "booking_status": "needs_clarification",
            "messages": [AIMessage(content=result["message"])],
        }

    summary = result["summary"]

    return {
        "booking_status": "cancelled",
        "cancel_suggestions": [],
        "messages": [
            AIMessage(
                content=(
                    "Cancellation confirmed.\n"
                    f"- Start: {summary['start_time']}\n"
                    f"- End: {summary['end_time']}\n"
                    f"- Slots released: {summary['cancelled_slots']}"
                )
            )
        ],
    }


def reschedule_confirmer_node(state: BookingAgentState) -> dict:
    intent = state.get("current_intent", {})
    old_start = intent.get("old_start_time")
    old_end = intent.get("old_end_time")
    new_start = intent.get("new_start_time")
    new_end = intent.get("new_end_time")
    details = intent.get("event_details")
    allow_conflict = bool(intent.get("allow_conflict"))

    if not old_start or not old_end or not new_start or not new_end:
        details_text = str(details or "").strip().lower()
        if details_text and details_text not in {"booking via ai assistant", "booking", "meeting"}:
            matches = [
                item for item in list_booked_events()
                if details_text in str(item.get("details", "")).lower()
            ]
            if len(matches) == 1:
                old_start = str(matches[0]["start_time"])
                old_end = str(matches[0]["end_time"])
            elif len(matches) > 1:
                options = "\n".join(
                    f"{i}. {item['start_time']} to {item['end_time']} ({item['details']})"
                    for i, item in enumerate(matches, start=1)
                )
                return {
                    "booking_status": "needs_clarification",
                    "reschedule_suggestions": matches,
                    "messages": [
                        AIMessage(
                            content=(
                                "I found multiple matching meetings to reschedule. Choose one first:\n"
                                f"{options}\n\n"
                                "Then tell me the new time (for example: move to tomorrow 5 PM)."
                            )
                        )
                    ],
                }

        if not old_start or not old_end:
            upcoming = list_booked_events()[:5]
            if upcoming:
                options = "\n".join(
                    f"{i}. {item['start_time']} to {item['end_time']} ({item['details']})"
                    for i, item in enumerate(upcoming, start=1)
                )
                return {
                    "booking_status": "needs_clarification",
                    "reschedule_suggestions": upcoming,
                    "messages": [
                        AIMessage(
                            content=(
                                "Please choose which confirmed booking to edit/reschedule:\n"
                                f"{options}\n\n"
                                "Reply with option number or booking name."
                            )
                        )
                    ],
                }

        return {
            "booking_status": "needs_clarification",
            "reschedule_suggestions": state.get("reschedule_suggestions", []),
            "messages": [
                AIMessage(
                    content=(
                        "Please provide both old and new times for reschedule. "
                        "Example: move tomorrow 4 PM meeting to Friday 5 PM."
                    )
                )
            ],
        }

    result = reschedule_meeting(
        old_start_time=str(old_start),
        old_end_time=str(old_end),
        new_start_time=str(new_start),
        new_end_time=str(new_end),
        details=str(details) if details else None,
    )

    if not result["success"]:
        if allow_conflict:
            record_conflict_event(
                start_time=str(new_start),
                end_time=str(new_end),
                details=str(details or "Booking via AI assistant"),
                action="reschedule",
            )
            return {
                "booking_status": "conflict",
                "messages": [
                    AIMessage(
                        content=(
                            "That reschedule still conflicts, so I logged it under conflicts as requested. "
                            "The original booking remains unchanged."
                        )
                    )
                ],
            }
        return {
            "booking_status": "conflict",
            "messages": [AIMessage(content=result["message"])],
        }

    summary = result["summary"]
    response = (
        "Rescheduled successfully.\n"
        f"- From: {summary['old_start_time']} to {summary['old_end_time']}\n"
        f"- To: {summary['new_start_time']} to {summary['new_end_time']}"
    )
    return {
        "booking_status": "rescheduled",
        "reschedule_suggestions": [],
        "messages": [AIMessage(content=response)],
    }


def reschedule_conflict_resolver_node(state: BookingAgentState) -> dict:
    intent = state.get("current_intent", {})
    target_start = str(intent.get("new_start_time") or intent.get("start_time") or "")
    target_end = str(intent.get("new_end_time") or intent.get("end_time") or "")

    if target_start and target_end:
        start_dt = datetime.strptime(target_start, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(target_end, "%Y-%m-%d %H:%M:%S")
        duration = max(30, int((end_dt - start_dt).total_seconds() // 60))
        suggestions = find_next_available_slots(target_start, duration, max_suggestions=3)
        if suggestions:
            text = "\n".join(
                f"{i}. {item['start']} to {item['end']}" for i, item in enumerate(suggestions, start=1)
            )
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "The new time conflicts with another booking.\n"
                            "Try one of these alternatives:\n"
                            f"{text}"
                        )
                    )
                ],
                "conflict_suggestions": [f"{item['start']} to {item['end']}" for item in suggestions],
            }

    return {
        "messages": [AIMessage(content="Reschedule target conflicts. Please share another preferred time.")]
    }


def availability_node(state: BookingAgentState) -> dict:
    intent = state.get("current_intent", {})
    start_time = intent.get("start_time")
    end_time = intent.get("end_time")

    if not start_time or not end_time:
        return {
            "booking_status": "needs_clarification",
            "messages": [
                AIMessage(
                    content="I need both start and end times to check availability."
                )
            ],
        }

    availability = check_availability(start_time, end_time)
    if availability["available"]:
        return {"booking_status": "pending"}

    if intent.get("allow_conflict"):
        return {
            "booking_status": "pending",
            "current_intent": {
                **intent,
                "conflict_record_only": True,
            },
        }

    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    duration = int((end_dt - start_dt).total_seconds() // 60)
    suggestions = find_next_available_slots(start_time, duration, max_suggestions=3)

    if not suggestions:
        suggestions = find_next_available_slots(
            _to_datetime_text(datetime.now()),
            duration,
            max_suggestions=3,
        )

    formatted_suggestions = [
        f"{item['start']} to {item['end']}" for item in suggestions
    ]

    return {
        "booking_status": "conflict",
        "conflict_suggestions": formatted_suggestions,
    }


def conflict_resolver_node(state: BookingAgentState, llm: BaseChatModel) -> dict:
    intent = state.get("current_intent", {})
    requested_window = (
        f"{intent.get('start_time', 'unknown')} to {intent.get('end_time', 'unknown')}"
    )
    suggestions = state.get("conflict_suggestions", [])

    if suggestions:
        option_lines = "\n".join(
            f"{idx}. {slot}" for idx, slot in enumerate(suggestions, start=1)
        )
        message = (
            f"That time is already booked: {requested_window}.\n\n"
            "Here are the next available options:\n"
            f"{option_lines}\n\n"
            "Reply with 'option 1', 'option 2', or paste a preferred time window. "
            "If you still want this conflicting slot, reply 'go with conflict'."
        )
    else:
        message = (
            f"That time is already booked: {requested_window}.\n"
            "I could not find nearby alternatives. Please share another preferred date/time."
        )

    return {"messages": [AIMessage(content=message)]}


def booking_confirmer_node(state: BookingAgentState) -> dict:
    intent = state.get("current_intent", {})
    start_time = intent.get("start_time")
    end_time = intent.get("end_time")
    details = intent.get("event_details", "Booking via AI assistant")
    conflict_record_only = bool(intent.get("conflict_record_only"))

    if not start_time or not end_time:
        return {
            "booking_status": "needs_clarification",
            "messages": [
                AIMessage(content="I could not confirm because the time window is incomplete.")
            ],
        }

    if conflict_record_only:
        record_conflict_event(
            start_time=str(start_time),
            end_time=str(end_time),
            details=str(details),
            action="book",
        )
        return {
            "booking_status": "conflict",
            "messages": [
                AIMessage(
                    content=(
                        "This slot conflicts with an existing booking, and I logged it under conflicts as requested. "
                        "No confirmed booking was created."
                    )
                )
            ],
        }

    result = book_meeting(start_time, end_time, details)

    if not result["success"]:
        return {
            "booking_status": "conflict",
            "messages": [AIMessage(content=result["message"])],
        }

    summary = result["summary"]
    confirmation_text = (
        "Booking confirmed.\n"
        f"- Start: {summary['start_time']}\n"
        f"- End: {summary['end_time']}\n"
        f"- Details: {summary['details']}\n"
        f"- Slots reserved: {summary['booked_slots']}"
    )

    return {
        "booking_status": "confirmed",
        "messages": [AIMessage(content=confirmation_text)],
    }


def route_after_availability(state: BookingAgentState) -> str:
    status = state.get("booking_status")
    if status == "pending":
        return "booking_confirmer"
    if status == "conflict":
        return "conflict_resolver"
    return END


def route_after_intent(state: BookingAgentState) -> str:
    status = state.get("booking_status")
    if status == "needs_clarification":
        return END

    action = str(state.get("current_intent", {}).get("action", "book"))
    if action == "cancel":
        return "cancellation_confirmer"
    if action == "reschedule":
        return "reschedule_confirmer"
    return "availability"


def route_after_reschedule(state: BookingAgentState) -> str:
    if state.get("booking_status") == "conflict":
        return "reschedule_conflict_resolver"
    return END


def create_booking_graph(
    model_name: str | None = None,
    provider: str | None = None,
):
    llm = _create_llm(model_name=model_name, provider=provider)

    workflow = StateGraph(BookingAgentState)
    workflow.add_node("intent_parser", lambda state: intent_parser_node(state, llm))
    workflow.add_node("availability", availability_node)
    workflow.add_node(
        "conflict_resolver", lambda state: conflict_resolver_node(state, llm)
    )
    workflow.add_node("booking_confirmer", booking_confirmer_node)
    workflow.add_node("cancellation_confirmer", cancellation_confirmer_node)
    workflow.add_node("reschedule_confirmer", reschedule_confirmer_node)
    workflow.add_node("reschedule_conflict_resolver", reschedule_conflict_resolver_node)

    workflow.add_edge(START, "intent_parser")
    workflow.add_conditional_edges(
        "intent_parser",
        route_after_intent,
        {
            "availability": "availability",
            "cancellation_confirmer": "cancellation_confirmer",
            "reschedule_confirmer": "reschedule_confirmer",
            END: END,
        },
    )

    # Conditional routing based on availability outcome:
    # - pending  -> booking_confirmer
    # - conflict -> conflict_resolver
    # - others   -> END
    workflow.add_conditional_edges(
        "availability",
        route_after_availability,
        {
            "booking_confirmer": "booking_confirmer",
            "conflict_resolver": "conflict_resolver",
            END: END,
        },
    )

    workflow.add_edge("booking_confirmer", END)
    workflow.add_edge("cancellation_confirmer", END)
    workflow.add_conditional_edges(
        "reschedule_confirmer",
        route_after_reschedule,
        {
            "reschedule_conflict_resolver": "reschedule_conflict_resolver",
            END: END,
        },
    )
    workflow.add_edge("reschedule_conflict_resolver", END)
    workflow.add_edge("conflict_resolver", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
