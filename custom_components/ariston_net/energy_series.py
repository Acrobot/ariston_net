"""Aggregate Ariston energy-consumption series by their reported period.

The p=1 series holds subdaily consumption buckets; p=3 and p=4 hold
daily and monthly values. The API does not provide bucket timestamps,
resolution or history length. Both subdaily parameters are configurable;
unsupported or misaligned layouts return None instead of estimated totals.

Use local time to locate today's portion of p=1 and cross-check the
previous day's buckets against the completed daily total.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from math import fsum, isfinite
from typing import Final, Literal

SEQUENCE_TYPE_KEY: Final = "k"
SEQUENCE_PERIOD_KEY: Final = "p"
SEQUENCE_VALUES_KEY: Final = "v"

Period = Literal["today", "current_month", "previous_month", "current_year"]


def _is_valid_kwh(value: object) -> bool:
    """Accept only finite, non-negative numeric energy readings."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        and value >= 0
    )


def _consumption_samples(
    sequences: Iterable[Mapping[str, object]], consumption_type: int, period_code: int
) -> list[float] | None:
    """Find a consumption series and validate all its samples."""
    for sequence in sequences:
        if (sequence.get(SEQUENCE_TYPE_KEY), sequence.get(SEQUENCE_PERIOD_KEY)) != (
            consumption_type,
            period_code,
        ):
            continue

        samples = sequence.get(SEQUENCE_VALUES_KEY)
        if not isinstance(samples, list) or not samples:
            return None
        if not all(_is_valid_kwh(sample) for sample in samples):
            return None

        return [float(sample) for sample in samples]

    return None


def _today_kwh(
    buckets: list[float] | None,
    days: list[float] | None,
    local_now: datetime,
    bucket_minutes: int,
    history_days: int,
) -> float | None:
    """Select today's buckets, validating yesterday against daily totals.

    A trailing sequence may include a provisional current/next bucket.
    The duration and total number of reporting days are caller-supplied;
    no particular device model is assumed.
    """
    if bucket_minutes <= 0 or 1440 % bucket_minutes:
        raise ValueError("bucket_minutes must evenly divide 24 hours")
    if history_days < 2:
        raise ValueError("history_days must be at least 2")

    buckets_per_day = 1440 // bucket_minutes
    if buckets is None or len(buckets) != history_days * buckets_per_day or not days:
        return None

    minute_of_day = local_now.hour * 60 + local_now.minute
    if local_now.second or local_now.microsecond:
        minute_of_day += 1
    today_slots = min(
        buckets_per_day,
        minute_of_day // bucket_minutes + int(minute_of_day % bucket_minutes != 0) + 1,
    )
    yesterday = buckets[-today_slots - buckets_per_day : -today_slots]
    if len(yesterday) != buckets_per_day:
        return None

    today = buckets[-today_slots:]
    if abs(fsum(yesterday) - days[-1]) > 0.1:
        return None
    if days[-1] == 0 and any(today):
        # A zero previous day cannot distinguish alternate bucket layouts.
        return None
    return fsum(today)


def consumption_kwh(
    sequences: Iterable[Mapping[str, object]],
    consumption_type: int,
    period: Period,
    local_now: datetime,
    bucket_minutes: int = 120,
    history_days: int = 2,
) -> float | None:
    """Return estimated consumption for the requested calendar period.

    Defaults describe a two-day history of 120-minute buckets; callers
    can supply another verified reporting profile. The cloud response
    alone cannot reliably determine the bucket duration or alignment.
    This is not a revenue-grade or lifetime meter. Missing or misaligned
    data return None rather than an invented total.
    """
    sequences = list(sequences)
    if period == "previous_month":
        months = _consumption_samples(sequences, consumption_type, 4)
        return round(months[-1], 3) if months is not None else None

    if period not in ("today", "current_month", "current_year"):
        raise ValueError(f"Unknown consumption period: {period}")

    hourly = _consumption_samples(sequences, consumption_type, 1)
    days = _consumption_samples(sequences, consumption_type, 3)
    today = _today_kwh(hourly, days, local_now, bucket_minutes, history_days)
    if today is None:
        return None
    if period == "today":
        return round(today, 3)

    finished_days = local_now.day - 1
    if finished_days:
        if days is None or len(days) < finished_days:
            return None
        month = fsum(days[-finished_days:]) + today
    else:
        month = today
    if period == "current_month":
        return round(month, 3)

    finished_months = local_now.month - 1
    if not finished_months:
        return round(month, 3)
    months = _consumption_samples(sequences, consumption_type, 4)
    if months is None or len(months) < finished_months:
        return None
    return round(fsum(months[-finished_months:]) + month, 3)
