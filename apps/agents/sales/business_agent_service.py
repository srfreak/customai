from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup

from apps.agents.sales import services
from apps.agents.sales.catalog import (
    resolve_industry_label,
    resolve_use_case_label,
    resolve_use_case_tone,
)


async def scrape_website_text(url: str, timeout: int = 12) -> str:
    def _fetch() -> str:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            texts = []
            for element in soup.find_all(["p", "li", "h1", "h2", "h3"]):
                text = element.get_text(separator=" ", strip=True)
                if text:
                    texts.append(text)
            return " ".join(texts)
        except Exception:
            return ""

    return await asyncio.to_thread(_fetch)


async def summarise_website(text: str, industry_id: str, use_case_id: str) -> str:
    if not text:
        return ""
    trimmed = re.sub(r"\s+", " ", text)[:6000]
    industry = resolve_industry_label(industry_id)
    use_case = resolve_use_case_label(use_case_id)
    prompt = (
        "You are a knowledge ingestion assistant. Summarise the following website copy to capture "
        f"messaging, value propositions, tone, and offerings relevant to {industry} {use_case}. "
        "Return a concise summary under 200 words.\n\nContent:\n"
        f"{trimmed}"
    )
    summary = await services.call_openai_chat(
        [
            {"role": "system", "content": "Provide concise business summaries."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    return summary.strip()


async def draft_strategy_from_context(
    *,
    industry_id: str,
    use_case_id: str,
    goal: str,
    persona_name: str,
    notes: Optional[str],
    knowledge_summary: Optional[str],
) -> Dict[str, Any]:
    industry = resolve_industry_label(industry_id)
    use_case = resolve_use_case_label(use_case_id)
    tone_hint = resolve_use_case_tone(use_case_id)
    context_lines = [
        f"Industry: {industry}",
        f"Business use case: {use_case}",
        f"Primary goal: {goal}",
        f"Desired persona tone: {tone_hint}",
    ]
    if notes:
        context_lines.append(f"Additional notes: {notes}")
    if knowledge_summary:
        context_lines.append(f"Website knowledge summary: {knowledge_summary}")

    prompt = "\n".join(context_lines)
    messages = [
        {
            "role": "system",
            "content": (
                "You are Scriza AI's strategy architect. Produce JSON describing a sales agent strategy."
                " Follow the StrategyPayload schema with persona, scripts, goals, objections, closing,"
                " fallback scenarios, and fallback policies. Ensure persona.name matches the provided"
                " agent name when applicable."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    completion = await services.call_openai_chat(messages, stream=False)
    return completion
