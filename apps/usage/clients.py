from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

import httpx

from core.config import settings

logger = logging.getLogger(__name__)


def _date_param(date: dt.date) -> str:
    return date.strftime("%Y-%m-%d")


async def fetch_twilio_usage(
    start_date: dt.date,
    end_date: dt.date,
    category: str = "calls",
) -> Dict[str, Any]:
    """
    Fetch usage records from Twilio Usage API.

    Docs: https://www.twilio.com/docs/usage/api/usage-record
    """
    account_sid = settings.TWILIO_ACCOUNT_SID
    auth_token = settings.TWILIO_AUTH_TOKEN
    if not (account_sid and auth_token):
        logger.warning("Twilio credentials not configured; returning empty usage.")
        return {"usage_records": []}

    url = (
        f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}"
        f"/Usage/Records/{category}.json"
    )
    params = {
        "StartDate": _date_param(start_date),
        "EndDate": _date_param(end_date),
    }
    async with httpx.AsyncClient(auth=(account_sid, auth_token), timeout=20) as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            logger.error(
                "Twilio usage API error: %s %s", resp.status_code, resp.text
            )
            return {"usage_records": []}
        return resp.json()


async def fetch_openai_usage(
    start_date: dt.date,
    end_date: dt.date,
) -> Dict[str, Any]:
    """
    Fetch usage metrics from OpenAI Usage API.

    Docs: https://platform.openai.com/docs/api-reference/usage
    """
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; OpenAI usage unavailable.")
        return {"data": []}

    url = "https://api.openai.com/v1/usage"
    params = {
        "start_date": _date_param(start_date),
        "end_date": _date_param(end_date),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            logger.error(
                "OpenAI usage API error: %s %s", resp.status_code, resp.text
            )
            return {"data": []}
        return resp.json()


async def fetch_elevenlabs_usage(
    start_date: dt.date,
    end_date: dt.date,
) -> Dict[str, Any]:
    """
    Fetch usage analytics from ElevenLabs.

    Docs: https://elevenlabs.io/docs/api-reference
    """
    api_key = settings.ELEVENLABS_API_KEY
    if not api_key:
        logger.warning("ELEVENLABS_API_KEY not set; ElevenLabs usage unavailable.")
        return {"usage": []}

    url = "https://api.elevenlabs.io/v1/usage"
    params = {
        "from_date": _date_param(start_date),
        "to_date": _date_param(end_date),
    }
    headers = {
        "xi-api-key": api_key,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            logger.error(
                "ElevenLabs usage API error: %s %s", resp.status_code, resp.text
            )
            return {"usage": []}
        return resp.json()
