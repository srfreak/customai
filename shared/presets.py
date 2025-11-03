from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from shared.cache import TTLCache

_TTL_SECONDS = 24 * 60 * 60
_cache = TTLCache(ttl_seconds=_TTL_SECONDS, max_items=256)


def _presets_path() -> Path:
    return Path(__file__).resolve().parent / "templates" / "agent_industry_presets.json"


def _load_presets_raw() -> Dict[str, Any]:
    path = _presets_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_presets() -> Dict[str, Any]:
    cached = _cache.get("presets")
    if cached is not None:
        return cached
    data = _load_presets_raw()
    _cache.set("presets", data)
    return data


def find_preset(industry: Optional[str], use_case: Optional[str]) -> Optional[Dict[str, Any]]:
    if not industry or not use_case:
        return None
    presets = load_presets()
    # Case-insensitive lookup
    for ind_key, ind_val in presets.items():
        if ind_key.strip().lower() == industry.strip().lower():
            if isinstance(ind_val, dict):
                for uc_key, preset in ind_val.items():
                    if uc_key.strip().lower() == use_case.strip().lower():
                        return preset if isinstance(preset, dict) else None
    return None


def save_presets(data: Dict[str, Any]) -> None:
    """Persist presets JSON to disk and refresh cache."""
    path = _presets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    _cache.set("presets", data)


def upsert_preset(industry: str, use_case: str, preset: Dict[str, Any]) -> Dict[str, Any]:
    """Create or update a preset for an industry/use_case."""
    data = load_presets() or {}
    ind = data.get(industry) or {}
    if not isinstance(ind, dict):
        ind = {}
    ind[use_case] = preset
    data[industry] = ind
    save_presets(data)
    return preset


def delete_preset(industry: str, use_case: str) -> bool:
    data = load_presets() or {}
    ind = data.get(industry)
    if not isinstance(ind, dict):
        return False
    removed = use_case in ind
    if removed:
        ind.pop(use_case, None)
        # Clean empty industry
        if not ind:
            data.pop(industry, None)
        else:
            data[industry] = ind
        save_presets(data)
    return removed

