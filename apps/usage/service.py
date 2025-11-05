from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.config import settings
from core.database import get_collection
from shared.cache import TTLCache

from .clients import (
    fetch_elevenlabs_usage,
    fetch_openai_usage,
    fetch_twilio_usage,
)


USAGE_CACHE = TTLCache(ttl_seconds=60, max_items=16)


def _convert_currency(amount_usd: float, target_currency: str) -> Dict[str, float]:
    target_currency = target_currency.upper()
    if target_currency == "USD":
        return {"amount": amount_usd, "currency": "USD"}
    if target_currency == "INR":
        converted = amount_usd * settings.EXCHANGE_RATE_USD_TO_INR
        return {"amount": converted, "currency": "INR"}
    # fallback: return USD
    return {"amount": amount_usd, "currency": "USD"}


def _cache_key(prefix: str, **params: Any) -> str:
    parts = [prefix] + [f"{k}:{v}" for k, v in sorted(params.items())]
    return "|".join(parts)


async def get_usage_summary(
    start_date: dt.date,
    end_date: dt.date,
    agent_id: Optional[str],
    tenant_id: Optional[str],
    currency: str,
) -> Dict[str, Any]:
    """Aggregate live usage across services + persisted call metrics."""
    cache_key = _cache_key(
        "summary",
        start=start_date,
        end=end_date,
        agent=agent_id or "",
        tenant=tenant_id or "",
        currency=currency,
    )
    cached = USAGE_CACHE.get(cache_key)
    if cached:
        return cached

    # External usage
    twilio_data = await fetch_twilio_usage(start_date, end_date, category="calls")
    openai_data = await fetch_openai_usage(start_date, end_date)
    eleven_data = await fetch_elevenlabs_usage(start_date, end_date)

    # Mongo call summaries
    calls_coll = get_collection("calls")
    query: Dict[str, Any] = {
        "created_at": {
            "$gte": dt.datetime.combine(start_date, dt.time.min),
            "$lte": dt.datetime.combine(end_date, dt.time.max),
        }
    }
    if agent_id:
        query["agent_id"] = agent_id
    if tenant_id:
        query["user_id"] = tenant_id
    cursor = calls_coll.find(query)
    calls = await cursor.to_list(length=2000)

    calls_total = len(calls)
    calls_completed = sum(1 for c in calls if c.get("status") == "completed")
    calls_failed = sum(1 for c in calls if c.get("status") == "failed")

    gpt_tokens = sum(c.get("tokens_used", 0) for c in calls)
    duration_seconds = sum(c.get("duration_seconds", 0.0) for c in calls)

    total_cost_usd = sum(c.get("total_cost", 0.0) for c in calls)
    costs_by_service = {
        "telephony_usd": sum(c.get("telephony_cost", 0.0) for c in calls),
        "gpt_usd": sum(c.get("gpt_cost", 0.0) for c in calls),
        "asr_usd": sum(c.get("asr_cost", 0.0) for c in calls),
        "tts_usd": sum(c.get("tts_cost", 0.0) for c in calls),
    }

    converted_total = _convert_currency(total_cost_usd, currency)
    converted_costs = {
        service: _convert_currency(value, currency)["amount"]
        for service, value in costs_by_service.items()
    }

    summary = {
        "date_range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "calls": {
            "total": calls_total,
            "completed": calls_completed,
            "failed": calls_failed,
            "duration_seconds": duration_seconds,
        },
        "gpt": {
            "tokens_used": gpt_tokens,
            "source": "openai",
            "usage_raw": openai_data,
        },
        "tts": {
            "usage_raw": eleven_data,
        },
        "telephony": {
            "usage_raw": twilio_data,
        },
        "costs": {
            "total": converted_total,
            "by_service": converted_costs,
            "currency": converted_total["currency"],
        },
        "calls_sample": [
            {
                "call_id": c.get("call_id"),
                "agent_id": c.get("agent_id"),
                "status": c.get("status"),
                "duration_seconds": c.get("duration_seconds"),
                "total_cost": _convert_currency(c.get("total_cost", 0.0), currency)["amount"],
            }
            for c in calls[:25]
        ],
    }

    USAGE_CACHE.set(cache_key, summary)
    return summary


async def get_usage_trend(
    start_date: dt.date,
    end_date: dt.date,
    agent_id: Optional[str],
    tenant_id: Optional[str],
    currency: str,
) -> Dict[str, Any]:
    """Return daily cost and call trends from Mongo summary collection."""
    summary_coll = get_collection("usage_summary")
    cursor = summary_coll.find(
        {
            "date": {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat(),
            },
            **({"agent_id": agent_id} if agent_id else {}),
            **({"tenant_id": tenant_id} if tenant_id else {}),
        }
    ).sort("date", 1)
    rows = await cursor.to_list(length=500)
    series = []
    for row in rows:
        usd_total = float(row.get("total_cost_usd", 0.0))
        converted = _convert_currency(usd_total, currency)
        series.append(
            {
                "date": row.get("date"),
                "total_cost": converted["amount"],
                "currency": converted["currency"],
                "calls": row.get("calls_total", 0),
                "asr_seconds": row.get("asr_seconds", 0.0),
                "gpt_tokens": row.get("gpt_tokens", 0),
                "tts_seconds": row.get("tts_seconds", 0.0),
            }
        )
    return {"series": series, "currency": currency}


async def get_call_usage_detail(call_id: str, currency: str) -> Dict[str, Any]:
    """Fetch per-call usage breakdown from Mongo."""
    calls_coll = get_collection("calls")
    call = await calls_coll.find_one({"call_id": call_id})
    if not call:
        return {"call_id": call_id, "error": "call_not_found"}

    return {
        "call_id": call_id,
        "agent_id": call.get("agent_id"),
        "status": call.get("status"),
        "duration_seconds": call.get("duration_seconds", 0.0),
        "asr": {
            "seconds": call.get("asr_seconds", 0.0),
            "model": call.get("asr_model"),
            "cost": _convert_currency(call.get("asr_cost", 0.0), currency)["amount"],
        },
        "gpt": {
            "input_tokens": call.get("gpt_input_tokens", 0),
            "output_tokens": call.get("gpt_output_tokens", 0),
            "model": call.get("gpt_model"),
            "cost": _convert_currency(call.get("gpt_cost", 0.0), currency)["amount"],
        },
        "tts": {
            "seconds": call.get("tts_seconds", 0.0),
            "model": call.get("tts_model"),
            "cache_hit": call.get("tts_cache_hit", False),
            "cost": _convert_currency(call.get("tts_cost", 0.0), currency)["amount"],
        },
        "telephony": {
            "cost": _convert_currency(call.get("telephony_cost", 0.0), currency)["amount"],
        },
        "total_cost": _convert_currency(call.get("total_cost", 0.0), currency),
        "currency": currency,
    }
