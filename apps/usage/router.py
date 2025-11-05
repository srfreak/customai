from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from core.auth import RoleChecker
from core.config import settings
from .service import (
    get_call_usage_detail,
    get_usage_summary,
    get_usage_trend,
)

router = APIRouter()


def _parse_date(value: Optional[str], default: dt.date) -> dt.date:
    if not value:
        return default
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date: {value}") from exc


@router.get("/usage/summary")
async def usage_summary(
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    agent_id: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    currency: Optional[str] = Query(None, description="USD or INR"),
    user: dict = Depends(RoleChecker(["admin"])),
):
    today = dt.date.today()
    start_date = _parse_date(start, today - dt.timedelta(days=6))
    end_date = _parse_date(end, today)
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start date must be before end date")
    target_currency = currency or settings.DEFAULT_USAGE_CURRENCY
    return await get_usage_summary(start_date, end_date, agent_id, tenant_id, target_currency)


@router.get("/usage/trend")
async def usage_trend(
    period: str = Query("30d", description="e.g., 7d, 30d, 90d"),
    agent_id: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    user: dict = Depends(RoleChecker(["admin"])),
):
    today = dt.date.today()
    try:
        days = int(period.rstrip("d"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid period format") from exc
    start_date = today - dt.timedelta(days=days - 1)
    end_date = today
    target_currency = currency or settings.DEFAULT_USAGE_CURRENCY
    return await get_usage_trend(start_date, end_date, agent_id, tenant_id, target_currency)


@router.get("/usage/call_detail")
async def usage_call_detail(
    call_id: str,
    currency: Optional[str] = Query(None),
    user: dict = Depends(RoleChecker(["admin", "user"])),
):
    target_currency = currency or settings.DEFAULT_USAGE_CURRENCY
    detail = await get_call_usage_detail(call_id, target_currency)
    if "error" in detail:
        raise HTTPException(status_code=404, detail="Call not found")
    return detail
