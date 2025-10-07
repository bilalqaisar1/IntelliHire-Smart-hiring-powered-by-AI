import re
from typing import Optional

def find_years_in_text(text: str) -> Optional[float]:
    """Extracts numeric years of experience from text."""
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years|yrs|year)", text, flags=re.IGNORECASE)
    if matches:
        try:
            return float(matches[0])
        except:
            return None
    return None
