from duckduckgo_search import DDGS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Any
import asyncio

# Ensure required NLTK datasets are downloaded
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng")


TRUSTED_DOMAINS: list[str] = [
    # ── GLOBAL WIRE SERVICES (most cited, highest editorial standards) ──
    "apnews.com",  # Associated Press
    "reuters.com",  # Reuters
    "afp.com",  # Agence France-Presse
    # ── MAJOR INTERNATIONAL BROADCASTERS ──
    "bbc.com",  # BBC News
    "bbc.co.uk",  # BBC News (UK)
    "dw.com",  # Deutsche Welle
    "france24.com",  # France 24
    "aljazeera.com",  # Al Jazeera English
    "rfi.fr",  # Radio France Internationale
    "abc.net.au",  # ABC Australia
    "cbc.ca",  # CBC (Canada)
    "rte.ie",  # RTÉ (Ireland)
    "nhk.or.jp",  # NHK (Japan)
    # ── US NEWSPAPERS & DIGITAL ──
    "nytimes.com",  # The New York Times
    "washingtonpost.com",  # The Washington Post
    "wsj.com",  # The Wall Street Journal
    "theguardian.com",  # The Guardian (US edition)
    "npr.org",  # NPR
    "pbs.org",  # PBS NewsHour
    "theatlantic.com",  # The Atlantic
    "politico.com",  # Politico
    "thehill.com",  # The Hill
    "csmonitor.com",  # Christian Science Monitor
    "usatoday.com",  # USA Today
    # ── BUSINESS & FINANCE ──
    "bloomberg.com",  # Bloomberg
    "ft.com",  # Financial Times
    "economist.com",  # The Economist
    "forbes.com",  # Forbes
    "businessinsider.com",  # Business Insider
    # ── UK & EUROPEAN OUTLETS ──
    "theguardian.com",  # The Guardian (UK)
    "telegraph.co.uk",  # The Daily Telegraph
    "thetimes.co.uk",  # The Times (UK)
    "independent.co.uk",  # The Independent
    "euronews.com",  # Euronews
    "spiegel.de",  # Der Spiegel (Germany)
    "lemonde.fr",  # Le Monde (France)
    "elpais.com",  # El País (Spain)
    # ── SCIENCE & SPECIALIZED ──
    "nature.com",  # Nature
    "science.org",  # Science Magazine
    "newscientist.com",  # New Scientist
    "scientificamerican.com",  # Scientific American
    "foreignpolicy.com",  # Foreign Policy
    "foreignaffairs.com",  # Foreign Affairs
    # ── ASIA-PACIFIC ──
    "straitstimes.com",  # The Straits Times (Singapore)
    "scmp.com",  # South China Morning Post
    "japantimes.co.jp",  # The Japan Times
    "koreatimes.co.kr",  # Korea Times
    "bangkokpost.com",  # Bangkok Post
    # ── FACT-CHECKERS (GLOBAL) ──
    "snopes.com",  # Snopes
    "factcheck.org",  # FactCheck.org
    "politifact.com",  # PolitiFact
    "fullfact.org",  # Full Fact (UK)
    # ──────────────────────────────────────
    # ── INDIAN NEWS OUTLETS ──
    # ──────────────────────────────────────
    # ── MOST BALANCED / EDITORIAL INDEPENDENCE ──
    "thehindu.com",  # The Hindu
    "indianexpress.com",  # The Indian Express
    "hindustantimes.com",  # Hindustan Times
    "livemint.com",  # Mint (business/economy)
    "business-standard.com",  # Business Standard
    # ── WIRE SERVICES (INDIA) ──
    "ptinews.com",  # Press Trust of India (PTI)
    "uniindia.com",  # United News of India (UNI)
    "ani.net.in",  # Asian News International (ANI)
    # ── DIGITAL & INVESTIGATIVE ──
    "scroll.in",  # Scroll.in
    "thewire.in",  # The Wire
    "theprint.in",  # The Print
    "thequint.com",  # The Quint
    "ndtv.com",  # NDTV
    "thenewsminute.com",  # The News Minute (South India)
    # ── BROADCAST ──
    "ddnews.gov.in",  # DD News (Doordarshan)
    "allindiaradio.gov.in",  # All India Radio
    # ── REGIONAL ENGLISH DAILIES ──
    "deccanherald.com",  # Deccan Herald
    "tribuneindia.com",  # The Tribune
    "telegraphindia.com",  # The Telegraph (Kolkata)
    "newindianexpress.com",  # The New Indian Express
    "thestatesman.com",  # The Statesman
    # ── FACT-CHECKERS (INDIA) ──
    "boomlive.in",  # BOOM Live
    "altnews.in",  # Alt News
    "factchecker.in",  # FactChecker.in
    "vishvasnews.com",  # Vishvas News
]


def extract_search_query(text: str, max_keywords: int = 10) -> str:
    """
    Uses POS tagging to extract the 'Who, What, and How Much'.
    """
    tokens = word_tokenize(text)
    tagged_words = nltk.pos_tag(tokens)
    stop_words = set(stopwords.words("english"))
    target_tags = {"NNP", "NNPS", "NN", "NNS", "CD"}

    keywords: list[str] = []
    seen = set()

    for word, tag in tagged_words:
        clean_word = word.lower()
        if tag in target_tags and clean_word not in stop_words and len(clean_word) > 1:
            if clean_word not in seen:
                seen.add(clean_word)
                keywords.append(word)

    return " ".join(keywords[:max_keywords])


async def verify_claims(text: str) -> dict[str, Any]:
    """
    Executes Pillar C logic: Validates the central claim via DuckDuckGo.
    """
    extracted_query: str = extract_search_query(text)

    if not extracted_query:
        return {
            "status": "failed_extraction",
            "corroborated": False,
            "trusted_matches": 0,
            "message": "Could not extract verifiable claims from the text.",
        }

    try:

        def search_duckduckgo():
            ddgs = DDGS()
            return list(ddgs.text(extracted_query, max_results=10))

        # Run synchronous search in thread pool to keep async interface
        results = await asyncio.to_thread(search_duckduckgo)

        # Filter the results locally to see if trusted domains reported on it
        match_count = 0
        for result in results:
            url = result.get("href", "").lower()
            if any(trusted_domain in url for trusted_domain in TRUSTED_DOMAINS):
                match_count += 1

        is_corroborated = match_count > 0

        return {
            "status": "success",
            "corroborated": is_corroborated,
            "trusted_matches": match_count,
            "query_used": extracted_query,
        }

    except Exception as e:
        return {
            "status": "api_error",
            "corroborated": False,
            "trusted_matches": 0,
            "message": f"Failed to connect to DuckDuckGo: {str(e)}",
        }
