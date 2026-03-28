"""FastAPI bridge for the LangGraph booking agent used by the React frontend."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import smtplib
import threading
import uuid
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from database import DATETIME_FMT, SLOT_MINUTES, get_connection
from database import init_db
from graph import create_booking_graph
from agent_tools import check_availability, find_next_available_slots, list_booked_events


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    booking_status: str
    conflict_suggestions: list[str]
    action_options: list[str]
    state: dict[str, Any]


class BookingItem(BaseModel):
    id: str
    title: str
    date: str
    time: str
    duration: str
    participants: list[str]
    status: str


class StatsResponse(BaseModel):
    total_bookings: int
    confirmed: int
    pending: int
    conflicts: int


class SummaryResponse(BaseModel):
    stats: StatsResponse
    bookings: list[BookingItem]


class NotificationItem(BaseModel):
    id: str
    type: str
    title: str
    message: str
    time: str
    read: bool


class NotificationListResponse(BaseModel):
    notifications: list[NotificationItem]
    unread_count: int


class MarkAllReadResponse(BaseModel):
    unread_count: int


class DismissResponse(BaseModel):
    ok: bool
    unread_count: int


class ImportCsvRequest(BaseModel):
    csv_content: str = Field(min_length=1)


class ImportCsvResponse(BaseModel):
    imported: int
    skipped: int
    errors: list[str]


class ActivityItem(BaseModel):
    id: int
    event_type: str
    title: str
    detail: str
    status: str
    created_at: str


class CalendarEvent(BaseModel):
    title: str
    start_time: str
    end_time: str
    status: str
    source: str = "local"


class EmailReportRequest(BaseModel):
    to_email: str = Field(min_length=5)
    subject: str | None = None
    provider: str | None = None
    smtp: dict[str, Any] | None = None
    resend: dict[str, Any] | None = None


class MeetingEmailReportRequest(BaseModel):
    to_email: str = Field(min_length=5)
    title: str = Field(min_length=1)
    date: str = Field(min_length=1)
    time: str = Field(min_length=1)
    duration: str = Field(min_length=1)
    status: str = Field(min_length=1)
    subject: str | None = None
    provider: str | None = None
    smtp: dict[str, Any] | None = None
    resend: dict[str, Any] | None = None


class EmailReportResponse(BaseModel):
    ok: bool
    message: str


class UpdateBookingRequest(BaseModel):
    title: str | None = None
    date: str | None = None
    time: str | None = None
    duration: str | None = None


class UpdateBookingResponse(BaseModel):
    ok: bool
    booking: BookingItem
    message: str


app = FastAPI(title="Booking Agent API", version="1.0.0")

allowed_origins = os.getenv(
    "AGENT_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
allowed_origin_regex = os.getenv(
    "AGENT_ALLOWED_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1|\[::1\]|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})(:\d+)?$",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins.split(",") if origin.strip()],
    allow_origin_regex=allowed_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
_model_name = (
    os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    if _provider == "ollama"
    else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
)
AGENT_GRAPH = create_booking_graph(provider=_provider, model_name=_model_name)


_notifications_lock = threading.Lock()
_notifications: list[dict[str, Any]] = [
    {
        "id": "1",
        "type": "success",
        "title": "Booking Agent Ready",
        "message": "The booking backend is connected and ready.",
        "time": "just now",
        "read": False,
    },
    {
        "id": "2",
        "type": "info",
        "title": "Tip",
        "message": "Try: Book a meeting tomorrow from 10:00 to 10:30.",
        "time": "just now",
        "read": False,
    },
]


def _fmt_date(value: datetime) -> str:
    return value.strftime("%Y-%m-%d")


def _fmt_time(value: datetime) -> str:
    return value.strftime("%I:%M %p")


def _fmt_duration_minutes(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes / 60
    if hours.is_integer():
        return f"{int(hours)} hr"
    return f"{hours:.1f} hrs"


def _fetch_grouped_bookings() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT start_datetime, end_datetime, event_details
            FROM calendar
            WHERE status = 'booked'
            ORDER BY start_datetime ASC
            """
        ).fetchall()

    grouped: list[dict[str, Any]] = []
    for row in rows:
        start = datetime.strptime(row["start_datetime"], DATETIME_FMT)
        end = datetime.strptime(row["end_datetime"], DATETIME_FMT)
        details = row["event_details"] or "Booking via AI assistant"

        if not grouped:
            grouped.append({"start": start, "end": end, "details": details})
            continue

        last = grouped[-1]
        if last["details"] == details and last["end"] == start:
            last["end"] = end
        else:
            grouped.append({"start": start, "end": end, "details": details})

    return grouped


def _booking_item_from_group(index: int, booking: dict[str, Any]) -> BookingItem:
    duration_minutes = int((booking["end"] - booking["start"]).total_seconds() // 60)
    return BookingItem(
        id=str(index),
        title=str(booking["details"]),
        date=_fmt_date(booking["start"]),
        time=_fmt_time(booking["start"]),
        duration=_fmt_duration_minutes(duration_minutes),
        participants=["AI Scheduled"],
        status="confirmed",
    )


def _fetch_bookings() -> list[BookingItem]:
    grouped = _fetch_grouped_bookings()
    if not grouped:
        return []

    bookings: list[BookingItem] = []
    for idx, booking in enumerate(grouped, start=1):
        bookings.append(_booking_item_from_group(idx, booking))

    # Append logged conflict requests so users can track intentionally conflicting choices.
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
        rows = conn.execute(
            """
            SELECT id, start_time, end_time, details
            FROM conflict_events
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()

    for row in rows:
        start = datetime.strptime(str(row["start_time"]), DATETIME_FMT)
        end = datetime.strptime(str(row["end_time"]), DATETIME_FMT)
        duration_minutes = int((end - start).total_seconds() // 60)
        bookings.append(
            BookingItem(
                id=f"conflict-{row['id']}",
                title=str(row["details"]),
                date=_fmt_date(start),
                time=_fmt_time(start),
                duration=_fmt_duration_minutes(max(30, duration_minutes)),
                participants=["AI Scheduled"],
                status="conflict",
            )
        )

    return bookings


def _build_stats(bookings: list[BookingItem]) -> StatsResponse:
    confirmed = len([b for b in bookings if b.status == "confirmed"])
    pending = len([b for b in bookings if b.status == "pending"])
    conflicts = len([b for b in bookings if b.status == "conflict"])
    return StatsResponse(
        total_bookings=len(bookings),
        confirmed=confirmed,
        pending=pending,
        conflicts=conflicts,
    )


def _unread_count(items: list[dict[str, Any]]) -> int:
    return len([item for item in items if not item.get("read", False)])


def _push_notification(kind: str, title: str, message: str) -> None:
    with _notifications_lock:
        next_id = str(max([int(item["id"]) for item in _notifications], default=0) + 1)
        _notifications.insert(
            0,
            {
                "id": next_id,
                "type": kind,
                "title": title,
                "message": message,
                "time": "just now",
                "read": False,
            },
        )


def _ensure_activity_schema() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                title TEXT NOT NULL,
                detail TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _log_activity(event_type: str, title: str, detail: str, status: str) -> None:
    _ensure_activity_schema()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO activity_log (event_type, title, detail, status, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_type, title, detail, status, datetime.now().strftime(DATETIME_FMT)),
        )
        conn.commit()


def _fetch_activity(limit: int = 30) -> list[ActivityItem]:
    _ensure_activity_schema()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, event_type, title, detail, status, created_at
            FROM activity_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [ActivityItem(**dict(row)) for row in rows]


def _fetch_calendar_events(start: datetime, end: datetime) -> list[CalendarEvent]:
    events: list[CalendarEvent] = []
    for event in list_booked_events():
        event_start = datetime.strptime(event["start_time"], DATETIME_FMT)
        event_end = datetime.strptime(event["end_time"], DATETIME_FMT)
        if event_end <= start or event_start >= end:
            continue
        events.append(
            CalendarEvent(
                title=event["details"],
                start_time=event["start_time"],
                end_time=event["end_time"],
                status="confirmed",
                source="local",
            )
        )

    # Include explicitly logged conflict requests so they are visible in calendar views.
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
        conflict_rows = conn.execute(
            """
            SELECT start_time, end_time, details
            FROM conflict_events
            ORDER BY id DESC
            LIMIT 200
            """
        ).fetchall()

    for row in conflict_rows:
        try:
            row_start = datetime.strptime(str(row["start_time"]), DATETIME_FMT)
            row_end = datetime.strptime(str(row["end_time"]), DATETIME_FMT)
        except ValueError:
            continue

        if row_end <= start or row_start >= end:
            continue

        events.append(
            CalendarEvent(
                title=f"{row['details']} (Conflict)",
                start_time=str(row["start_time"]),
                end_time=str(row["end_time"]),
                status="conflict",
                source="local",
            )
        )

    # Mark conflict only for true overlap cases.
    for i, item in enumerate(events):
        a_start = datetime.strptime(item.start_time, DATETIME_FMT)
        a_end = datetime.strptime(item.end_time, DATETIME_FMT)
        for j, other in enumerate(events):
            if i == j:
                continue
            b_start = datetime.strptime(other.start_time, DATETIME_FMT)
            b_end = datetime.strptime(other.end_time, DATETIME_FMT)
            if a_start < b_end and a_end > b_start:
                item.status = "conflict"
                break

    return events


def _is_valid_email(value: str) -> bool:
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", value.strip()))


def _deliver_email_or_fallback(
    to_email: str,
    subject: str,
    body: str,
    smtp_override: dict[str, Any] | None = None,
    resend_override: dict[str, Any] | None = None,
    provider_preference: str | None = None,
) -> EmailReportResponse:
    smtp_override = smtp_override or {}

    smtp_host = str(smtp_override.get("host") or os.getenv("SMTP_HOST", "")).strip()
    smtp_port_text = str(smtp_override.get("port") or os.getenv("SMTP_PORT", "587")).strip()
    smtp_user = str(smtp_override.get("user") or os.getenv("SMTP_USER", "")).strip()
    smtp_password = str(smtp_override.get("password") or os.getenv("SMTP_PASSWORD", "")).strip()
    smtp_from = str(smtp_override.get("from_email") or os.getenv("SMTP_FROM", smtp_user)).strip()
    use_tls_value = smtp_override.get("use_tls")
    if use_tls_value is None:
        smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes"}
    else:
        smtp_use_tls = bool(use_tls_value)

    using_custom_smtp = bool(smtp_override)

    try:
        smtp_port = int(smtp_port_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="SMTP_PORT must be a valid integer") from exc

    if using_custom_smtp and (not smtp_host or not smtp_from):
        raise HTTPException(
            status_code=400,
            detail="Custom SMTP requires at least host and from_email.",
        )

    # Direct email API path (no SMTP) if configured.
    resend_override = resend_override or {}
    resend_api_key = str(resend_override.get("api_key") or os.getenv("RESEND_API_KEY", "")).strip()
    resend_from_email = str(resend_override.get("from_email") or os.getenv("RESEND_FROM_EMAIL", "")).strip()

    provider = (provider_preference or "").strip().lower()

    if resend_override and (not resend_api_key or not resend_from_email):
        raise HTTPException(
            status_code=400,
            detail="Resend configuration requires api_key and from_email.",
        )

    can_use_resend = resend_api_key and resend_from_email
    should_use_resend = provider == "resend" or (provider not in {"smtp", "resend"} and ((not smtp_host) or bool(resend_override)))

    if should_use_resend and can_use_resend:
        payload = {
            "from": resend_from_email,
            "to": [to_email],
            "subject": subject,
            "text": body,
        }
        req = urllib_request.Request(
            "https://api.resend.com/emails",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {resend_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=20) as response:
                if int(getattr(response, "status", 200)) >= 400:
                    raise HTTPException(status_code=500, detail="Email API rejected the request")
        except urllib_error.HTTPError as exc:
            raw_detail = exc.read().decode("utf-8", errors="ignore")
            parsed_message = raw_detail
            try:
                payload = json.loads(raw_detail)
                if isinstance(payload, dict):
                    parsed_message = str(
                        payload.get("message")
                        or payload.get("error")
                        or payload.get("name")
                        or raw_detail
                    )
            except Exception:
                pass

            lowered = parsed_message.lower()
            if "1010" in lowered or "resend.dev" in lowered:
                parsed_message = (
                    "Resend rejected the request. Ensure the sender is valid for your account. "
                    "If using onboarding@resend.dev, you may only send to your own account email. "
                    "To send broadly, verify your domain in Resend and use a sender from that domain."
                )
            elif "invalid api key" in lowered or "invalid_api_key" in lowered:
                parsed_message = "Resend API key is invalid. Generate a new key in Resend dashboard and retry."
            elif "domain" in lowered and "not verified" in lowered:
                parsed_message = (
                    "Sender domain is not verified in Resend. Verify your domain and use a sender address from it."
                )

            raise HTTPException(status_code=500, detail=f"Email API failed: {parsed_message}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Email API failed: {exc}") from exc

        return EmailReportResponse(ok=True, message=f"Report sent to {to_email}.")

    if provider == "smtp" and (not smtp_host or not smtp_from):
        raise HTTPException(
            status_code=400,
            detail="SMTP provider selected but SMTP host/from_email is missing. Fill SMTP settings or configure SMTP_* env vars.",
        )

    if not smtp_host or not smtp_from:
        reports_dir = os.getenv("BOOKING_REPORTS_DIR", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fallback_path = os.path.join(reports_dir, f"email-report-{timestamp}.txt")
        with open(fallback_path, "w", encoding="utf-8") as report_file:
            report_file.write(f"To: {to_email}\n")
            report_file.write(f"Subject: {subject}\n\n")
            report_file.write(body)

        return EmailReportResponse(
            ok=True,
            message=(
                "SMTP is not configured. Report was saved locally at "
                f"{fallback_path}. Configure SMTP_* or RESEND_API_KEY + RESEND_FROM_EMAIL for direct email delivery."
            ),
        )

    msg = EmailMessage()
    msg["From"] = smtp_from
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            if smtp_use_tls:
                server.starttls()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.send_message(msg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to send email report: {exc}") from exc

    return EmailReportResponse(ok=True, message=f"Report sent to {to_email}.")


def _parse_duration_minutes(text: str) -> int:
    lowered = text.strip().lower()
    min_match = re.search(r"(\d{1,3})\s*(min|mins|minute|minutes)", lowered)
    if min_match:
        return int(min_match.group(1))

    hr_match = re.search(r"(\d{1,2})(?:\.(\d))?\s*(h|hr|hrs|hour|hours)", lowered)
    if hr_match:
        whole = int(hr_match.group(1))
        decimal = hr_match.group(2)
        if decimal:
            return max(SLOT_MINUTES, int(float(f"{whole}.{decimal}") * 60))
        return max(SLOT_MINUTES, whole * 60)

    if lowered.isdigit():
        return max(SLOT_MINUTES, int(lowered))

    return SLOT_MINUTES


def _parse_start_datetime(date_text: str, time_text: str) -> datetime:
    date_text = date_text.strip()
    time_text = time_text.strip().upper()

    date_formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y"]
    time_formats = ["%H:%M", "%I:%M %p", "%I %p"]

    parsed_date = None
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_text, fmt).date()
            break
        except ValueError:
            continue
    if parsed_date is None:
        raise ValueError(f"Unsupported date format: {date_text}")

    parsed_time = None
    for fmt in time_formats:
        try:
            parsed_time = datetime.strptime(time_text, fmt).time()
            break
        except ValueError:
            continue
    if parsed_time is None:
        raise ValueError(f"Unsupported time format: {time_text}")

    return datetime.combine(parsed_date, parsed_time)


def _upsert_booked_window(start_dt: datetime, end_dt: datetime, details: str) -> None:
    with get_connection() as conn:
        cursor = start_dt
        while cursor < end_dt:
            slot_end = cursor + timedelta(minutes=SLOT_MINUTES)
            start_text = cursor.strftime(DATETIME_FMT)
            end_text = slot_end.strftime(DATETIME_FMT)

            existing = conn.execute(
                """
                SELECT id
                FROM calendar
                WHERE start_datetime = ? AND end_datetime = ?
                """,
                (start_text, end_text),
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE calendar
                    SET status = 'booked', event_details = ?
                    WHERE id = ?
                    """,
                    (details, existing["id"]),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO calendar (start_datetime, end_datetime, status, event_details)
                    VALUES (?, ?, 'booked', ?)
                    """,
                    (start_text, end_text, details),
                )

            cursor = slot_end

        conn.commit()


def _remove_booked_window(start_dt: datetime, end_dt: datetime, details: str | None = None) -> int:
    start_text = start_dt.strftime(DATETIME_FMT)
    end_text = end_dt.strftime(DATETIME_FMT)
    with get_connection() as conn:
        params: list[Any] = [start_text, end_text]
        sql = """
            UPDATE calendar
            SET status = 'free', event_details = NULL
            WHERE start_datetime >= ?
              AND end_datetime <= ?
              AND status = 'booked'
        """
        if details:
            sql += " AND event_details = ?"
            params.append(details)

        cur = conn.execute(sql, tuple(params))
        conn.commit()
        return cur.rowcount


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    thread_id = payload.thread_id or str(uuid.uuid4())

    try:
        result_state = AGENT_GRAPH.invoke(
            {
                "messages": [HumanMessage(content=payload.message)],
            },
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

    reply = "I could not generate a response."
    for message in reversed(result_state.get("messages", [])):
        if isinstance(message, AIMessage):
            reply = message.content if isinstance(message.content, str) else str(message.content)
            break

    booking_status = str(result_state.get("booking_status", "pending"))
    conflict_suggestions = [str(x) for x in result_state.get("conflict_suggestions", [])]
    cancel_suggestions = result_state.get("cancel_suggestions", [])
    reschedule_suggestions = result_state.get("reschedule_suggestions", [])
    current_intent = result_state.get("current_intent", {})

    suggestion_source: list[Any] = []
    if conflict_suggestions:
        suggestion_source = conflict_suggestions
    elif isinstance(cancel_suggestions, list) and cancel_suggestions:
        suggestion_source = cancel_suggestions
    elif isinstance(reschedule_suggestions, list) and reschedule_suggestions:
        suggestion_source = reschedule_suggestions

    action_options: list[str] = []
    for item in suggestion_source:
        if isinstance(item, str):
            action_options.append(item)
            continue

        if isinstance(item, dict):
            start = str(item.get("start_time", "")).strip()
            end = str(item.get("end_time", "")).strip()
            details = str(item.get("details", "Booking")).strip() or "Booking"
            if start and end:
                action_options.append(f"{start} to {end} ({details})")
                continue

        action_options.append(str(item))

    if booking_status == "confirmed":
        start_time = str(current_intent.get("start_time", "scheduled slot"))
        _push_notification(
            "success",
            "Meeting Confirmed",
            f"Booking confirmed for {start_time}.",
        )
        _log_activity("booking_created", "Meeting confirmed", f"Booking confirmed for {start_time}.", "confirmed")
    elif booking_status == "cancelled":
        start_time = str(current_intent.get("start_time", "requested slot"))
        _push_notification(
            "info",
            "Meeting Cancelled",
            f"Booking cancelled for {start_time}.",
        )
        _log_activity("meeting_cancelled", "Meeting cancelled", f"Booking cancelled for {start_time}.", "cancelled")
    elif booking_status == "rescheduled":
        _log_activity("meeting_rescheduled", "Meeting rescheduled", reply, "rescheduled")
    elif booking_status == "conflict":
        _push_notification(
            "warning",
            "Scheduling Conflict",
            "Requested slot is unavailable. Alternatives are available.",
        )
        _log_activity("conflict_detected", "Conflict detected", "Requested slot is unavailable.", "conflict")

    return ChatResponse(
        reply=reply,
        thread_id=thread_id,
        booking_status=booking_status,
        conflict_suggestions=conflict_suggestions,
        action_options=action_options,
        state={"current_intent": current_intent},
    )


@app.get("/api/bookings", response_model=list[BookingItem])
def get_bookings() -> list[BookingItem]:
    return _fetch_bookings()


@app.get("/api/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    bookings = _fetch_bookings()
    return _build_stats(bookings)


@app.get("/api/summary", response_model=SummaryResponse)
def get_summary() -> SummaryResponse:
    bookings = _fetch_bookings()
    return SummaryResponse(stats=_build_stats(bookings), bookings=bookings)


@app.get("/api/activity", response_model=list[ActivityItem])
def get_activity() -> list[ActivityItem]:
    return _fetch_activity()


@app.get("/api/calendar/events", response_model=list[CalendarEvent])
def get_calendar_events(start: str, end: str) -> list[CalendarEvent]:
    try:
        start_dt = datetime.strptime(start, DATETIME_FMT)
        end_dt = datetime.strptime(end, DATETIME_FMT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="start/end must use YYYY-MM-DD HH:MM:SS") from exc

    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end must be after start")

    return _fetch_calendar_events(start_dt, end_dt)


@app.patch("/api/bookings/{booking_id}", response_model=UpdateBookingResponse)
def update_booking(booking_id: str, payload: UpdateBookingRequest) -> UpdateBookingResponse:
    grouped = _fetch_grouped_bookings()
    if not grouped:
        raise HTTPException(status_code=404, detail="No bookings available to update")

    try:
        index = int(booking_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid booking id") from exc

    if index < 1 or index > len(grouped):
        raise HTTPException(status_code=404, detail="Booking not found")

    current = grouped[index - 1]
    current_start: datetime = current["start"]
    current_end: datetime = current["end"]
    current_title = str(current["details"])
    current_duration_minutes = int((current_end - current_start).total_seconds() // 60)

    raw_title = payload.title.strip() if payload.title else current_title
    raw_date = payload.date.strip() if payload.date else current_start.strftime("%Y-%m-%d")
    raw_time = payload.time.strip() if payload.time else current_start.strftime("%H:%M")
    raw_duration = payload.duration.strip() if payload.duration else str(current_duration_minutes)

    if not raw_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")

    try:
        new_start = _parse_start_datetime(raw_date, raw_time)
        new_duration_minutes = _parse_duration_minutes(raw_duration)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    new_end = new_start + timedelta(minutes=new_duration_minutes)

    if new_end <= new_start:
        raise HTTPException(status_code=400, detail="Invalid duration")

    same_window = new_start == current_start and new_end == current_end
    same_title = raw_title == current_title

    if same_window and same_title:
        return UpdateBookingResponse(
            ok=True,
            booking=_booking_item_from_group(index, current),
            message="No changes detected.",
        )

    if same_window:
        _upsert_booked_window(new_start, new_end, raw_title)
    else:
        released = _remove_booked_window(current_start, current_end, current_title)
        if released <= 0:
            raise HTTPException(status_code=409, detail="Booking could not be updated. Please refresh and retry.")

        new_start_text = new_start.strftime(DATETIME_FMT)
        new_end_text = new_end.strftime(DATETIME_FMT)
        availability = check_availability(new_start_text, new_end_text)
        if not availability["available"]:
            _upsert_booked_window(current_start, current_end, current_title)
            suggestions = find_next_available_slots(new_start_text, new_duration_minutes, max_suggestions=3)
            raise HTTPException(
                status_code=409,
                detail={
                    "message": f"Cannot move booking: {availability['reason']}",
                    "suggestions": [f"{item['start']} to {item['end']}" for item in suggestions],
                },
            )

        _upsert_booked_window(new_start, new_end, raw_title)

    updated_group = {
        "start": new_start,
        "end": new_end,
        "details": raw_title,
    }
    _log_activity(
        "meeting_updated",
        "Meeting updated",
        f"Updated booking '{current_title}' to '{raw_title}' on {new_start.strftime('%Y-%m-%d %H:%M')}",
        "confirmed",
    )
    return UpdateBookingResponse(ok=True, booking=_booking_item_from_group(index, updated_group), message="Booking updated.")


def _build_email_report_body() -> str:
    bookings = _fetch_bookings()
    stats = _build_stats(bookings)

    lines = [
        "Booking Agent Report",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Summary",
        f"- Total: {stats.total_bookings}",
        f"- Confirmed: {stats.confirmed}",
        f"- Pending: {stats.pending}",
        f"- Conflicts: {stats.conflicts}",
        "",
        "Bookings",
    ]

    if not bookings:
        lines.append("- No bookings yet")
    else:
        for item in bookings:
            lines.append(
                f"- {item.title} | {item.date} {item.time} | {item.duration} | {item.status}"
            )

    return "\n".join(lines)


@app.post("/api/reports/email", response_model=EmailReportResponse)
def email_report(payload: EmailReportRequest) -> EmailReportResponse:
    to_email = payload.to_email.strip()
    if not _is_valid_email(to_email):
        raise HTTPException(status_code=400, detail="Invalid recipient email address")
    report_body = _build_email_report_body()
    report_subject = payload.subject or "Booking Agent Summary Report"

    return _deliver_email_or_fallback(
        to_email=to_email,
        subject=report_subject,
        body=report_body,
        smtp_override=payload.smtp,
        resend_override=payload.resend,
        provider_preference=payload.provider,
    )


@app.post("/api/reports/meeting-email", response_model=EmailReportResponse)
def email_meeting_report(payload: MeetingEmailReportRequest) -> EmailReportResponse:
    to_email = payload.to_email.strip()
    if not _is_valid_email(to_email):
        raise HTTPException(status_code=400, detail="Invalid recipient email address")

    subject = payload.subject or f"Meeting Report: {payload.title}"
    body = "\n".join(
        [
            "Booking Agent Meeting Report",
            "",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Title: {payload.title}",
            f"Date: {payload.date}",
            f"Time: {payload.time}",
            f"Duration: {payload.duration}",
            f"Status: {payload.status}",
        ]
    )

    return _deliver_email_or_fallback(
        to_email=to_email,
        subject=subject,
        body=body,
        smtp_override=payload.smtp,
        resend_override=payload.resend,
        provider_preference=payload.provider,
    )


@app.get("/api/notifications", response_model=NotificationListResponse)
def get_notifications() -> NotificationListResponse:
    with _notifications_lock:
        items = [NotificationItem(**n) for n in _notifications]
        unread = _unread_count(_notifications)
    return NotificationListResponse(notifications=items, unread_count=unread)


@app.post("/api/notifications/mark-all-read", response_model=MarkAllReadResponse)
def mark_notifications_read() -> MarkAllReadResponse:
    with _notifications_lock:
        for item in _notifications:
            item["read"] = True
        unread = _unread_count(_notifications)
    return MarkAllReadResponse(unread_count=unread)


@app.delete("/api/notifications/{notification_id}", response_model=DismissResponse)
def dismiss_notification(notification_id: str) -> DismissResponse:
    with _notifications_lock:
        idx = next(
            (i for i, item in enumerate(_notifications) if item["id"] == notification_id),
            None,
        )
        if idx is None:
            raise HTTPException(status_code=404, detail="Notification not found")
        _notifications.pop(idx)
        unread = _unread_count(_notifications)
    return DismissResponse(ok=True, unread_count=unread)


@app.post("/api/bookings/import-csv", response_model=ImportCsvResponse)
def import_bookings_csv(payload: ImportCsvRequest) -> ImportCsvResponse:
    reader = csv.DictReader(io.StringIO(payload.csv_content))
    required = {"title", "date", "time", "duration"}
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV must include a header row")

    header_fields = {name.strip().lower() for name in reader.fieldnames}
    if not required.issubset(header_fields):
        raise HTTPException(
            status_code=400,
            detail="CSV headers must include: title,date,time,duration",
        )

    imported = 0
    skipped = 0
    errors: list[str] = []

    for idx, row in enumerate(reader, start=2):
        if not row:
            skipped += 1
            continue

        normalized = {str(k).strip().lower(): (v or "").strip() for k, v in row.items()}
        status = normalized.get("status", "confirmed").lower()
        if status and status not in {"confirmed", "booked", "pending", "conflict", "cancelled", "canceled"}:
            skipped += 1
            errors.append(f"Line {idx}: unsupported status '{status}'")
            continue

        if status in {"pending", "conflict"}:
            skipped += 1
            continue

        title = normalized.get("title", "")
        date_text = normalized.get("date", "")
        time_text = normalized.get("time", "")
        duration_text = normalized.get("duration", "")

        if not title or not date_text or not time_text or not duration_text:
            skipped += 1
            errors.append(f"Line {idx}: missing required field values")
            continue

        try:
            start_dt = _parse_start_datetime(date_text, time_text)
            duration_minutes = _parse_duration_minutes(duration_text)
            end_dt = start_dt + timedelta(minutes=duration_minutes)
            start_text = start_dt.strftime(DATETIME_FMT)
            end_text = end_dt.strftime(DATETIME_FMT)

            if status in {"cancelled", "canceled"}:
                released = _remove_booked_window(start_dt, end_dt, title)
                if released > 0:
                    imported += 1
                else:
                    skipped += 1
                    errors.append(f"Line {idx}: no matching booking found to cancel")
                continue

            availability = check_availability(start_text, end_text)
            if not availability["available"]:
                skipped += 1
                errors.append(f"Line {idx}: {availability['reason']}")
                continue

            _upsert_booked_window(start_dt, end_dt, title)
            imported += 1
        except Exception as exc:  # pragma: no cover
            skipped += 1
            errors.append(f"Line {idx}: {exc}")

    if imported > 0:
        _push_notification(
            "success",
            "CSV Imported",
            f"Imported {imported} booking(s) from CSV.",
        )

    return ImportCsvResponse(imported=imported, skipped=skipped, errors=errors[:10])
