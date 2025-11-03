from __future__ import annotations

from typing import Dict, List

INDUSTRY_OPTIONS: List[Dict[str, str]] = [
    {"id": "retail_ecommerce", "label": "Retail & E-commerce"},
    {"id": "healthcare_medical", "label": "Healthcare & Medical"},
    {"id": "finance_banking", "label": "Finance & Banking"},
    {"id": "real_estate", "label": "Real Estate"},
    {"id": "education_training", "label": "Education & Training"},
    {"id": "hospitality_travel", "label": "Hospitality & Travel"},
    {"id": "automotive", "label": "Automotive"},
    {"id": "professional_services", "label": "Professional Services"},
    {"id": "technology_software", "label": "Technology & Software"},
    {"id": "government_public", "label": "Government & Public"},
    {"id": "food_beverage", "label": "Food & Beverage"},
    {"id": "manufacturing", "label": "Manufacturing"},
    {"id": "fitness_wellness", "label": "Fitness & Wellness"},
    {"id": "legal_services", "label": "Legal Services"},
    {"id": "non_profit", "label": "Non-Profit"},
    {"id": "media_entertainment", "label": "Media & Entertainment"},
    {"id": "other", "label": "Other"},
]

USE_CASE_OPTIONS: List[Dict[str, str]] = [
    {"id": "customer_support", "label": "Customer Support"},
    {"id": "outbound_sales", "label": "Outbound Sales"},
    {"id": "learning_development", "label": "Learning and Development"},
    {"id": "scheduling", "label": "Scheduling"},
    {"id": "lead_qualification", "label": "Lead Qualification"},
    {"id": "answering_service", "label": "Answering Service"},
    {"id": "property_search", "label": "Property Search"},
    {"id": "viewing_appointments", "label": "Viewing Appointments"},
    {"id": "market_information", "label": "Market Information"},
    {"id": "mortgage_guidance", "label": "Mortgage Guidance"},
    {"id": "listing_information", "label": "Listing Information"},
    {"id": "other", "label": "Other"},
]

USE_CASE_TONE_HINTS: Dict[str, str] = {
    "customer_support": "empathetic",
    "outbound_sales": "assertive",
    "learning_development": "informative",
    "scheduling": "efficient",
    "lead_qualification": "confident",
    "answering_service": "friendly",
    "property_search": "consultative",
    "viewing_appointments": "organized",
    "market_information": "analytical",
    "mortgage_guidance": "reassuring",
    "listing_information": "professional",
    "other": "adaptive",
}

INDUSTRY_LOOKUP = {item["id"]: item["label"] for item in INDUSTRY_OPTIONS}
USE_CASE_LOOKUP = {item["id"]: item["label"] for item in USE_CASE_OPTIONS}


def resolve_industry_label(industry_id: str) -> str:
    return INDUSTRY_LOOKUP.get(industry_id, industry_id.replace("_", " ").title())


def resolve_use_case_label(use_case_id: str) -> str:
    return USE_CASE_LOOKUP.get(use_case_id, use_case_id.replace("_", " ").title())


def resolve_use_case_tone(use_case_id: str) -> str:
    return USE_CASE_TONE_HINTS.get(use_case_id, "adaptive")
