"""LangGraph state schema and booking-related tools."""

from __future__ import annotations

from datetime import datetime, timedelta
from math import ceil
from typing import Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from database import DATETIME_FMT, SLOT_MINUTES, get_connection
from typing import TypedDict, List, Dict, Any


class BookingAgentState(TypedDict, total=False):
    messages: list
    current_intent: dict
    booking_status: str
    conflict_suggestions: list[str]
    cancel_suggestions: list[dict]
    reschedule_suggestions: list[dict]


def _ensure_conflict_log_schema() -> None:
	with get_connection() as conn:
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS conflict_events (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				start_time TEXT NOT NULL,
				end_time TEXT NOT NULL,
				details TEXT NOT NULL,
				action TEXT NOT NULL,
				created_at TEXT NOT NULL
			)
			"""
		)
		conn.commit()


def record_conflict_event(
	start_time: str,
	end_time: str,
	details: str,
	action: str,
) -> None:
	"""Persist a conflict attempt when user chooses to proceed with conflicts."""
	_ensure_conflict_log_schema()
	with get_connection() as conn:
		conn.execute(
			"""
			INSERT INTO conflict_events (start_time, end_time, details, action, created_at)
			VALUES (?, ?, ?, ?, ?)
			""",
			(start_time, end_time, details or "Booking via AI assistant", action, _to_datetime_str(datetime.now())),
		)
		conn.commit()


def _parse_datetime(value: str) -> datetime:
	return datetime.strptime(value, DATETIME_FMT)


def _to_datetime_str(value: datetime) -> str:
	return value.strftime(DATETIME_FMT)


def check_availability(start_time: str, end_time: str) -> dict[str, Any]:
	"""Check if every slot in a requested interval is free."""
	start_dt = _parse_datetime(start_time)
	end_dt = _parse_datetime(end_time)

	if end_dt <= start_dt:
		return {
			"available": False,
			"reason": "End time must be later than start time.",
		}

	with get_connection() as conn:
		conflict_count_row = conn.execute(
			"""
			SELECT COUNT(*) AS conflict_count
			FROM calendar
			WHERE status = 'booked'
			  AND start_datetime < ?
			  AND end_datetime > ?
			""",
			(end_time, start_time),
		).fetchone()

		if conflict_count_row and conflict_count_row["conflict_count"] > 0:
			return {
				"available": False,
				"reason": "Requested window overlaps an existing booking.",
			}

		free_rows = conn.execute(
			"""
			SELECT start_datetime
			FROM calendar
			WHERE status = 'free'
			  AND start_datetime >= ?
			  AND end_datetime <= ?
			ORDER BY start_datetime
			""",
			(start_time, end_time),
		).fetchall()

	free_slot_starts = {_parse_datetime(row["start_datetime"]) for row in free_rows}

	cursor = start_dt
	while cursor < end_dt:
		if cursor not in free_slot_starts:
			return {
				"available": False,
				"reason": "One or more requested slots are unavailable.",
			}
		cursor += timedelta(minutes=SLOT_MINUTES)

	return {
		"available": True,
		"reason": "All requested slots are free.",
		"slot_count": int((end_dt - start_dt).total_seconds() // (SLOT_MINUTES * 60)),
	}


def list_booked_events() -> list[dict[str, str]]:
	"""Return grouped booked windows from slot records."""
	with get_connection() as conn:
		rows = conn.execute(
			"""
			SELECT start_datetime, end_datetime, event_details
			FROM calendar
			WHERE status = 'booked'
			ORDER BY start_datetime ASC
			"""
		).fetchall()

	events: list[dict[str, str]] = []
	for row in rows:
		start = row["start_datetime"]
		end = row["end_datetime"]
		details = row["event_details"] or "Booking via AI assistant"
		if events and events[-1]["details"] == details and events[-1]["end_time"] == start:
			events[-1]["end_time"] = end
		else:
			events.append(
				{
					"start_time": start,
					"end_time": end,
					"details": details,
				}
			)

	return events


def book_meeting(start_time: str, end_time: str, details: str) -> dict[str, Any]:
	"""Book all slots in a requested interval if currently free."""
	availability = check_availability(start_time, end_time)
	if not availability["available"]:
		return {
			"success": False,
			"message": f"Could not complete booking: {availability['reason']}",
		}

	with get_connection() as conn:
		cur = conn.execute(
			"""
			UPDATE calendar
			SET status = 'booked', event_details = ?
			WHERE start_datetime >= ?
			  AND end_datetime <= ?
			  AND status = 'free'
			""",
			(details, start_time, end_time),
		)
		conn.commit()

	if cur.rowcount <= 0:
		return {
			"success": False,
			"message": "No slots were updated. Another booking may have occurred.",
		}

	return {
		"success": True,
		"message": "Meeting confirmed.",
		"summary": {
			"start_time": start_time,
			"end_time": end_time,
			"details": details,
			"booked_slots": cur.rowcount,
		},
	}


def cancel_meeting(
	start_time: str,
	end_time: str,
	details: str | None = None,
) -> dict[str, Any]:
	"""Cancel a booked interval by marking slots back to free."""
	start_dt = _parse_datetime(start_time)
	end_dt = _parse_datetime(end_time)
	if end_dt <= start_dt:
		return {
			"success": False,
			"message": "Cancellation window is invalid.",
		}

	with get_connection() as conn:
		params: list[Any] = [start_time, end_time]
		sql = """
			UPDATE calendar
			SET status = 'free', event_details = NULL
			WHERE start_datetime >= ?
			  AND end_datetime <= ?
			  AND status = 'booked'
		"""
		if details:
			sql += " AND LOWER(event_details) = LOWER(?)"
			params.append(details)

		cur = conn.execute(sql, tuple(params))

		# If exact detail filtering fails, retry by time window only.
		if cur.rowcount <= 0 and details:
			cur = conn.execute(
				"""
				UPDATE calendar
				SET status = 'free', event_details = NULL
				WHERE start_datetime >= ?
				  AND end_datetime <= ?
				  AND status = 'booked'
				""",
				(start_time, end_time),
			)
		conn.commit()

	if cur.rowcount <= 0:
		return {
			"success": False,
			"message": "No matching booked meeting found to cancel.",
		}

	return {
		"success": True,
		"message": "Meeting cancelled.",
		"summary": {
			"start_time": start_time,
			"end_time": end_time,
			"cancelled_slots": cur.rowcount,
			"details": details or "Any details",
		},
	}


def reschedule_meeting(
	old_start_time: str,
	old_end_time: str,
	new_start_time: str,
	new_end_time: str,
	details: str | None = None,
) -> dict[str, Any]:
	"""Move a booked meeting window to a new free window."""
	old_start = _parse_datetime(old_start_time)
	old_end = _parse_datetime(old_end_time)
	new_start = _parse_datetime(new_start_time)
	new_end = _parse_datetime(new_end_time)

	if old_end <= old_start or new_end <= new_start:
		return {"success": False, "message": "Invalid time windows for rescheduling."}

	if old_start_time == new_start_time and old_end_time == new_end_time:
		return {"success": False, "message": "Old and new meeting windows are identical."}

	with get_connection() as conn:
		params: list[Any] = [old_start_time, old_end_time]
		exists_sql = """
			SELECT COUNT(*) AS count
			FROM calendar
			WHERE start_datetime >= ?
			  AND end_datetime <= ?
			  AND status = 'booked'
		"""
		if details:
			exists_sql += " AND LOWER(event_details) = LOWER(?)"
			params.append(details)

		exists = conn.execute(exists_sql, tuple(params)).fetchone()
		if (not exists or int(exists["count"]) <= 0) and details:
			exists = conn.execute(
				"""
				SELECT COUNT(*) AS count
				FROM calendar
				WHERE start_datetime >= ?
				  AND end_datetime <= ?
				  AND status = 'booked'
				""",
				(old_start_time, old_end_time),
			).fetchone()
		if not exists or int(exists["count"]) <= 0:
			return {"success": False, "message": "Original meeting not found."}

	availability = check_availability(new_start_time, new_end_time)
	if not availability["available"]:
		return {
			"success": False,
			"message": f"Cannot reschedule: {availability['reason']}",
		}

	with get_connection() as conn:
		# Release old slots.
		release_params: list[Any] = [old_start_time, old_end_time]
		release_sql = """
			UPDATE calendar
			SET status = 'free', event_details = NULL
			WHERE start_datetime >= ?
			  AND end_datetime <= ?
			  AND status = 'booked'
		"""
		if details:
			release_sql += " AND LOWER(event_details) = LOWER(?)"
			release_params.append(details)
		release_cur = conn.execute(release_sql, tuple(release_params))

		if release_cur.rowcount <= 0 and details:
			release_cur = conn.execute(
				"""
				UPDATE calendar
				SET status = 'free', event_details = NULL
				WHERE start_datetime >= ?
				  AND end_datetime <= ?
				  AND status = 'booked'
				""",
				(old_start_time, old_end_time),
			)

		# Reserve new slots.
		reserve_details = details or "Booking via AI assistant"
		reserve_cur = conn.execute(
			"""
			UPDATE calendar
			SET status = 'booked', event_details = ?
			WHERE start_datetime >= ?
			  AND end_datetime <= ?
			  AND status = 'free'
			""",
			(reserve_details, new_start_time, new_end_time),
		)
		conn.commit()

	if reserve_cur.rowcount <= 0:
		return {
			"success": False,
			"message": "Reschedule failed while reserving new time window.",
		}

	return {
		"success": True,
		"message": "Meeting rescheduled.",
		"summary": {
			"old_start_time": old_start_time,
			"old_end_time": old_end_time,
			"new_start_time": new_start_time,
			"new_end_time": new_end_time,
			"details": details or "Booking via AI assistant",
		},
	}


def find_next_available_slots(
	start_time: str,
	duration_minutes: int,
	max_suggestions: int = 3,
) -> list[dict[str, str]]:
	"""Find the next contiguous free windows matching the requested duration."""
	required_slots = max(1, ceil(duration_minutes / SLOT_MINUTES))

	with get_connection() as conn:
		rows = conn.execute(
			"""
			SELECT start_datetime
			FROM calendar
			WHERE status = 'free'
			  AND start_datetime >= ?
			ORDER BY start_datetime
			""",
			(start_time,),
		).fetchall()

	free_starts = [_parse_datetime(row["start_datetime"]) for row in rows]
	free_start_set = set(free_starts)

	suggestions: list[dict[str, str]] = []
	seen_starts: set[datetime] = set()

	for candidate_start in free_starts:
		if candidate_start in seen_starts:
			continue

		contiguous = True
		for slot_idx in range(required_slots):
			slot_time = candidate_start + timedelta(minutes=SLOT_MINUTES * slot_idx)
			if slot_time not in free_start_set:
				contiguous = False
				break

		if contiguous:
			candidate_end = candidate_start + timedelta(
				minutes=SLOT_MINUTES * required_slots
			)
			suggestions.append(
				{
					"start": _to_datetime_str(candidate_start),
					"end": _to_datetime_str(candidate_end),
				}
			)
			seen_starts.add(candidate_start)

		if len(suggestions) >= max_suggestions:
			break

	return suggestions
