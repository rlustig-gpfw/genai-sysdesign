import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from langchain_core.tools import tool

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback only
    ZoneInfo = None  # type: ignore

try:
    # Optional: handle months/years precisely if available
    from dateutil.relativedelta import relativedelta  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    relativedelta = None  # type: ignore


def _get_zoneinfo(tz_name: Optional[str]) -> timezone:
    if not tz_name:
        return timezone.utc
    if tz_name.upper() in {"UTC", "Z"}:
        return timezone.utc
    if ZoneInfo is None:
        # Fallback gracefully when zoneinfo is unavailable
        return timezone.utc
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return timezone.utc


def _parse_iso_datetime(value: str, assumed_tz: timezone) -> datetime:
    # Handle common 'Z' suffix
    cleaned = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except Exception as exc:
        raise ValueError(f"Invalid ISO datetime: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=assumed_tz)
    return dt.astimezone(timezone.utc)


def _format_parts(dt: datetime, target_tz: timezone) -> Dict[str, str]:
    local = dt.astimezone(target_tz)
    return {
        "datetime_iso": local.isoformat(),
        "date": local.strftime("%Y-%m-%d"),
        "time": local.strftime("%H:%M"),
        "time_seconds": local.strftime("%H:%M:%S"),
        "weekday": local.strftime("%a"),
        "timezone": getattr(target_tz, "key", "UTC") if hasattr(target_tz, "key") else ("UTC" if target_tz == timezone.utc else str(target_tz)),
    }


def _compute_next_occurrence(weekday: str, time_hhmm: str, after_dt_utc: datetime, tz: timezone) -> datetime:
    weekday_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    if weekday not in weekday_map:
        raise ValueError("Invalid weekday. Use Mon|Tue|Wed|Thu|Fri|Sat|Sun")

    hours, minutes = [int(x) for x in time_hhmm.split(":", 1)]

    # Work in local tz, then return UTC
    local_after = after_dt_utc.astimezone(tz)
    target = local_after.replace(hour=hours, minute=minutes, second=0, microsecond=0)

    days_ahead = (weekday_map[weekday] - local_after.weekday()) % 7
    if days_ahead == 0 and target <= local_after:
        days_ahead = 7
    target = target + timedelta(days=days_ahead)
    return target.astimezone(timezone.utc)


def _apply_offset(dt_utc: datetime, offset: Dict[str, int]) -> datetime:
    years = int(offset.get("years", 0))
    months = int(offset.get("months", 0))
    weeks = int(offset.get("weeks", 0))
    days = int(offset.get("days", 0))
    hours = int(offset.get("hours", 0))
    minutes = int(offset.get("minutes", 0))

    if (years != 0 or months != 0) and relativedelta is None:
        # Fall back by approximating months/years (not exact for month lengths/leap years)
        # Prefer informing via minimal approximation: convert years/months to days rough estimate
        approx_days = years * 365 + months * 30
        dt_utc = dt_utc + timedelta(days=approx_days)
    elif years != 0 or months != 0:
        dt_utc = dt_utc + relativedelta(years=years, months=months)  # type: ignore[arg-type]

    dt_utc = dt_utc + timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes)
    return dt_utc


class DateTime:
    """A tool to manage date and time."""

    @tool
    @staticmethod
    def get_current_time() -> str:
        """Get the current time."""
        return datetime.now().strftime("%H:%M:%S")

    
class DateTimeTool:
    """Natural-language date/time question handler.

    This single function covers common date/time intents without requiring
    structured JSON. It recognizes queries like:
    - What's the current date/time (optionally in a timezone)?
    - What's the date tomorrow/yesterday?
    - Convert 2025-01-05 14:30 UTC to PST
    - How many days until 2025-12-25?
    - Difference between 2025-01-01 and 2025-02-01 in days
    - Next Friday at 3pm (optionally in a timezone)
    - Add/subtract offsets: in 3 days, 2 weeks from now, 5 hours ago
    """

    _TZ_ALIAS = {
        "utc": "UTC",
        "gmt": "UTC",
        "pt": "America/Los_Angeles",
        "pst": "America/Los_Angeles",
        "pdt": "America/Los_Angeles",
        "et": "America/New_York",
        "est": "America/New_York",
        "edt": "America/New_York",
        "ct": "America/Chicago",
        "cst": "America/Chicago",
        "cdt": "America/Chicago",
        "mt": "America/Denver",
        "mst": "America/Denver",
        "mdt": "America/Denver",
    }

    @staticmethod
    def _normalize_tz_name(text: str) -> Optional[str]:
        key = text.strip().lower()
        if key in DateTimeTool._TZ_ALIAS:
            return DateTimeTool._TZ_ALIAS[key]
        # pass through likely IANA tz names like Europe/London
        if "/" in text:
            return text
        # If appears like uppercase abbreviation, try alias
        if key.isalpha() and len(key) in (2, 3):
            return DateTimeTool._TZ_ALIAS.get(key)
        return None

    @staticmethod
    def _extract_timezone(question_lc: str) -> Optional[str]:
        # patterns: "in PST", "to PST", "in Europe/London"
        match = re.search(r"\b(?:in|to)\s+([A-Za-z_\/]+)\b", question_lc)
        if match:
            return DateTimeTool._normalize_tz_name(match.group(1))
        return None

    @staticmethod
    def _parse_hhmm_ampm(text: str) -> Optional[str]:
        # returns HH:MM 24h
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
        if not m:
            return None
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3)
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    @staticmethod
    def _parse_iso_snippet(text: str) -> Optional[str]:
        # Flexible ISO-like snippet detection
        m = re.search(
            r"(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?(?:Z|[+-]\d{2}:\d{2})?)",
            text,
        )
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _weekday_to_abbrev(text: str) -> Optional[str]:
        days = {
            "monday": "Mon",
            "tuesday": "Tue",
            "wednesday": "Wed",
            "thursday": "Thu",
            "friday": "Fri",
            "saturday": "Sat",
            "sunday": "Sun",
        }
        for name, abbr in days.items():
            if name in text:
                return abbr
        return None

    @tool
    @staticmethod
    def evaluate(question: str, current_time_iso: Optional[str] = None, default_timezone: str = "UTC") -> str:
        """Answer a natural-language date/time question.

        Args:
            question: Natural-language question about date/time.
            current_time_iso: Optional ISO timestamp to use as "now".
            default_timezone: Fallback timezone when not specified.

        Returns:
            A concise answer including ISO date/time and timezone when relevant.
        """
        try:
            now_utc = (
                _parse_iso_datetime(current_time_iso.replace("Z", "+00:00"), timezone.utc)
                if current_time_iso
                else datetime.now(timezone.utc)
            )
        except Exception:
            now_utc = datetime.now(timezone.utc)

        default_tz = _get_zoneinfo(default_timezone or "UTC")
        q_raw = question or ""
        q = q_raw.strip()
        q_lc = q.lower()

        # Determine target timezone if specified
        tz_name = DateTimeTool._extract_timezone(q_lc) or default_timezone
        tz = _get_zoneinfo(tz_name)

        # 1) Current date/time
        if re.search(r"\b(current|now|right now)\b", q_lc) and ("time" in q_lc or "date" in q_lc):
            parts = _format_parts(now_utc, tz)
            if "time" in q_lc and "date" not in q_lc:
                return f"Current time: {parts['time_seconds']} ({parts['timezone']})"
            if "date" in q_lc and "time" not in q_lc:
                return f"Today is {parts['date']} ({parts['timezone']})"
            return f"Now: {parts['datetime_iso']} ({parts['timezone']})"

        # 2) Relative day keywords
        if "tomorrow" in q_lc:
            dt = (now_utc + timedelta(days=1))
            parts = _format_parts(dt, tz)
            return f"Tomorrow: {parts['date']} ({parts['timezone']})"
        if "yesterday" in q_lc:
            dt = (now_utc - timedelta(days=1))
            parts = _format_parts(dt, tz)
            return f"Yesterday: {parts['date']} ({parts['timezone']})"

        # 3) Add/subtract offsets: in N units, N units from now, N units ago
        m = re.search(r"\b(in|within)\s+(\d+)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b", q_lc)
        if not m:
            m = re.search(r"\b(\d+)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+(from now|later)\b", q_lc)
            sign = +1
        else:
            sign = +1
        if not m:
            m = re.search(r"\b(\d+)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+ago\b", q_lc)
            sign = -1
        if m:
            value = int(m.group(2 if m.lastindex and m.lastindex >= 2 else 1))
            unit = m.group(3 if m.lastindex and m.lastindex >= 3 else 2)
            unit = unit.rstrip("s")
            offset = {unit + "s": sign * value}
            dt = _apply_offset(now_utc, offset)
            parts = _format_parts(dt, tz)
            return f"Result: {parts['datetime_iso']} ({parts['timezone']})"

        # 4) Conversion: convert <iso-like> to <tz>
        if "convert" in q_lc or "to" in q_lc or "in" in q_lc:
            iso_snippet = DateTimeTool._parse_iso_snippet(q)
            target_tz_name = DateTimeTool._extract_timezone(q_lc) or tz_name
            if iso_snippet and target_tz_name:
                dt = _parse_iso_datetime(iso_snippet, default_tz)
                ttz = _get_zoneinfo(target_tz_name)
                parts = _format_parts(dt, ttz)
                return f"Converted: {parts['datetime_iso']} ({parts['timezone']})"

        # 5) Difference: days/hours/minutes between two times or until X
        if "until" in q_lc or "till" in q_lc:
            iso_snippet = DateTimeTool._parse_iso_snippet(q)
            if iso_snippet:
                dt = _parse_iso_datetime(iso_snippet, default_tz)
                delta = dt - now_utc
                days = int(delta.total_seconds() // 86400)
                return f"Time until {dt.astimezone(tz).isoformat()}: {days} days"
        if "difference" in q_lc or "between" in q_lc:
            matches = re.findall(
                r"(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?(?:Z|[+-]\d{2}:\d{2})?)",
                q,
            )
            if len(matches) >= 2:
                a = _parse_iso_datetime(matches[0], default_tz)
                b = _parse_iso_datetime(matches[1], default_tz)
                delta = abs(b - a)
                days = int(delta.total_seconds() // 86400)
                hours = int((delta.total_seconds() % 86400) // 3600)
                minutes = int((delta.total_seconds() % 3600) // 60)
                return f"Difference: {days} days, {hours} hours, {minutes} minutes"

        # 6) Next weekday at time
        if "next" in q_lc:
            weekday_abbrev = DateTimeTool._weekday_to_abbrev(q_lc)
            if weekday_abbrev:
                time_24 = DateTimeTool._parse_hhmm_ampm(q_lc) or "00:00"
                next_dt_utc = _compute_next_occurrence(weekday_abbrev, time_24, now_utc, tz)
                parts = _format_parts(next_dt_utc, tz)
                return f"Next {weekday_abbrev}: {parts['datetime_iso']} ({parts['timezone']})"

        # 7) If asking simply for time in a place: "what time is it in PST/Europe/London"
        if re.search(r"what(?:'s| is)?\s+the\s+time\s+in\s+", q_lc) or re.search(r"time\s+in\s+", q_lc):
            parts = _format_parts(now_utc, tz)
            return f"Current time: {parts['time_seconds']} ({parts['timezone']})"

        # Fallback: provide current datetime in default/target tz
        parts = _format_parts(now_utc, tz)
        return f"Now: {parts['datetime_iso']} ({parts['timezone']})"