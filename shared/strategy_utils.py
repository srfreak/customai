from __future__ import annotations

from typing import Any, Dict, List, Optional


def summarize_strategy_payload(strategy: Optional[Dict[str, Any]]) -> Optional[str]:
    """Produce a compact textual summary for LLM grounding from a strategy payload.

    Mirrors the logic used by agents and services to avoid duplication.
    """
    if not strategy:
        return None
    payload = (
        strategy.get("payload")
        if isinstance(strategy, dict) and "payload" in strategy
        else strategy
    )
    if not isinstance(payload, dict):
        return None
    lines: List[str] = []
    scripts = payload.get("scripts") if isinstance(payload.get("scripts"), dict) else {}
    for key in ("greeting", "pitch", "faqs", "objections", "closing"):
        value = payload.get(key) or scripts.get(key)
        if not value:
            continue
        if isinstance(value, dict):
            lines.append(f"{key.title()}: " + "; ".join(f"{k}: {v}" for k, v in value.items()))
        elif isinstance(value, (list, tuple)):
            lines.append(f"{key.title()}: " + "; ".join(str(item) for item in value))
        else:
            lines.append(f"{key.title()}: {value}")

    # Product summary
    prods = payload.get("product_details") or payload.get("products") or []
    if isinstance(prods, list) and prods:
        try:
            sample = prods[0]
            name = sample.get("name")
            value = sample.get("value") or sample.get("benefits")
            if name:
                lines.append(f"Product: {name}")
            if value:
                if isinstance(value, list):
                    lines.append("Benefits: " + "; ".join(map(str, value)))
                else:
                    lines.append(f"Value: {value}")
        except Exception:
            pass

    # Audience and business info
    audience = payload.get("target_audience") or {}
    if isinstance(audience, dict):
        pains = audience.get("pain_points")
        if pains:
            pains_str = ", ".join(map(str, pains)) if isinstance(pains, list) else str(pains)
            lines.append(f"Audience Pain Points: {pains_str}")
    biz = payload.get("business_info") or {}
    if isinstance(biz, dict) and biz.get("company_name"):
        tagline = biz.get("tagline") or ""
        lines.append(f"Company: {biz.get('company_name')} — {tagline}")
    if payload.get("persona") and isinstance(payload["persona"], dict):
        voice_hint = payload["persona"].get("voice_id")
        if voice_hint:
            lines.append(f"Preferred Voice ID: {voice_hint}")

    return "\n".join(lines) if lines else None

