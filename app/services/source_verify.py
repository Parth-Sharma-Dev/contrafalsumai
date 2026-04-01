import whois
import asyncio
from datetime import datetime
from urllib.parse import urlparse
from typing import Any


HIGH_RISK_TLDS: set[str] = {".xyz", ".top", ".click", ".news", ".biz", ".info"}


def extract_root_domain(url: str) -> str:
    """Extracts the clean root domain (e.g., 'example.com') from a full URL."""
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc or parsed_url.path.split("/")[0]
        domain = domain.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


async def get_whois_data(domain: str) -> Any:
    """Wraps the blocking python-whois library in an async thread."""
    try:
        return await asyncio.to_thread(whois.whois, domain)
    except Exception:
        return None


async def analyze_domain(url: str) -> dict[str, Any]:
    """
    Executes TLD analysis and Domain Age.
    """
    domain: str = extract_root_domain(url)

    if not domain:
        return {
            "status": "invalid_url",
            "domain": None,
            "is_high_risk_tld": False,
            "domain_age_days": None,
            "high_risk_age": False,
        }

    # 1. TLD Analysis using the local set
    is_high_risk_tld: bool = any(domain.endswith(tld) for tld in HIGH_RISK_TLDS)

    # 2. Domain Age Check via WHOIS
    domain_age_days: int | None = None
    high_risk_age: bool = False

    whois_info = await get_whois_data(domain)

    if whois_info and whois_info.creation_date:
        creation_date = whois_info.creation_date

        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        if isinstance(creation_date, datetime):
            creation_date_naive = creation_date.replace(tzinfo=None)
            age_timedelta = datetime.now() - creation_date_naive
            domain_age_days = age_timedelta.days

            # Flag if domain is less than 6 months old
            if domain_age_days < 180:
                high_risk_age = True

    return {
        "status": "success",
        "domain": domain,
        "is_high_risk_tld": is_high_risk_tld,
        "domain_age_days": domain_age_days,
        "high_risk_age": high_risk_age,
    }
