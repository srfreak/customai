from typing import Dict, Any, Optional
import json
import re

def is_valid_email(email: str) -> bool:
    """Check if email is valid"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_valid_phone_number(phone: str) -> bool:
    """Check if phone number is valid"""
    pattern = r'^\+?1?-?\.?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$'
    return re.match(pattern, phone) is not None

def sanitize_input(text: str) -> str:
    """Sanitize user input"""
    # Remove any HTML/JS
    sanitized = re.sub(r'<[^>]*script[^\>]*>', '', text, flags=re.IGNORECASE)
    # Remove extra whitespace
    sanitized = ' '.join(sanitized.split())
    return sanitized

def format_phone_number(phone: str) -> str:
    """Format phone number"""
    # Remove all non-digits
    digits = re.sub(r'\D', '', phone)
    
    # Check if it's a valid US number
    if len(digits) == 10:
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
    elif len(digits) == 11 and digits[0] == '1':
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:11]}"
    
    # Return as is if not a standard US number
    return phone
