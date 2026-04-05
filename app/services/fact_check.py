from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

import dotenv

dotenv.load_dotenv()

import nltk
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from ddgs import DDGS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
for _resource, _pkg in [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("corpora/stopwords", "stopwords"),
]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_pkg)

# ── Trusted domains ───────────────────────────────────────────────────────────
TRUSTED_DOMAINS: list[str] = [
    "apnews.com",
    "reuters.com",
    "afp.com",
    "bbc.com",
    "bbc.co.uk",
    "dw.com",
    "france24.com",
    "aljazeera.com",
    "rfi.fr",
    "abc.net.au",
    "cbc.ca",
    "rte.ie",
    "nhk.or.jp",
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "theguardian.com",
    "npr.org",
    "pbs.org",
    "theatlantic.com",
    "politico.com",
    "thehill.com",
    "csmonitor.com",
    "usatoday.com",
    "bloomberg.com",
    "ft.com",
    "economist.com",
    "forbes.com",
    "businessinsider.com",
    "telegraph.co.uk",
    "thetimes.co.uk",
    "independent.co.uk",
    "euronews.com",
    "spiegel.de",
    "lemonde.fr",
    "elpais.com",
    "nature.com",
    "science.org",
    "newscientist.com",
    "scientificamerican.com",
    "foreignpolicy.com",
    "foreignaffairs.com",
    "straitstimes.com",
    "scmp.com",
    "japantimes.co.jp",
    "koreatimes.co.kr",
    "bangkokpost.com",
    "snopes.com",
    "factcheck.org",
    "politifact.com",
    "fullfact.org",
    "thehindu.com",
    "indianexpress.com",
    "hindustantimes.com",
    "livemint.com",
    "business-standard.com",
    "ptinews.com",
    "uniindia.com",
    "ani.net.in",
    "scroll.in",
    "thewire.in",
    "theprint.in",
    "thequint.com",
    "ndtv.com",
    "thenewsminute.com",
    "ddnews.gov.in",
    "allindiaradio.gov.in",
    "deccanherald.com",
    "tribuneindia.com",
    "telegraphindia.com",
    "newindianexpress.com",
    "thestatesman.com",
    "boomlive.in",
    "altnews.in",
    "factchecker.in",
    "vishvasnews.com",
]


GEMINI_MODEL = "gemini-2.5-flash"

# ── Gemini client (lazy-initialised) ─────────────────────────────────────────
_gemini_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini_client


# ── Shared retry helpers ──────────────────────────────────────────────────────


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _parse_retry_delay(e: Exception) -> float:
    try:
        match = re.search(r"retryDelay.*?(\d+(?:\.\d+)?)\s*s", str(e))
        if match:
            return float(match.group(1)) + 1.0
    except Exception:
        pass
    return 5.0


def _call_gemini(
    prompt: str, system: str, max_tokens: int = 100, max_retries: int = 3
) -> str | None:
    """
    Central Gemini call with 429 retry logic.
    Returns the response text, or None on failure.
    """
    client = _get_client()
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt.strip(),
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                ),
            )
            return response.text.strip()
        except ClientError as e:
            if _is_rate_limit_error(e):
                delay = _parse_retry_delay(e)
                print(
                    f"[gemini] Rate limited. Waiting {delay:.1f}s "
                    f"(attempt {attempt}/{max_retries})..."
                )
                time.sleep(delay)
            else:
                print(f"[gemini] Non-recoverable error: {e}")
                return None
        except Exception as e:
            print(f"[gemini] Unexpected error: {e}")
            return None
    print(f"[gemini] All {max_retries} retries exhausted.")
    return None


# STEP 1 — Extract search query

_QUERY_EXTRACTION_PROMPT = """\
You are a search-query distillation expert for a fake-news verification system.

Given a piece of text (which may be real or fabricated news), extract the single
most verifiable, specific factual claim at its core and convert it into a precise
DuckDuckGo search query of 8–15 words that preserves as much context as possible.

Rules:
- The query MUST include BOTH the subject (who/what) AND the event/action (what happened).
  A query that is only a person's name or an organisation's name is ALWAYS wrong.
- Capture the KEY CLAIM in full detail: who did what, where, when, to whom, using what.
- Include ALL specific named entities: organisations, people, technologies, places,
  product names, policy names, bill names, locations.
- For claims about death/health/arrest/resignation of a person, ALWAYS include
  the action word (dead, died, death, arrested, resigned, etc.) in the query.
- Include numbers, dates, quantities if they are central to the claim.
- Omit adjectives like "shocking", "revolutionary", "greatest".
- Omit filler verbs like "said", "claimed", "reported".
- DO NOT include synonyms, alternatives, or multiple queries.
- Output ONLY the raw query string. No quotes. No explanation. No punctuation.

Examples
--------
Input : "Ajit Pawar is dead"
Output: Ajit Pawar dead death confirmed news

Input : "Scientists at MIT have cured cancer using a new AI drug they call NeuroHeal"
Output: MIT scientists cured cancer new AI drug NeuroHeal breakthrough

Input : "The Reserve Bank of India has approved Quantum Rupee cryptocurrency for official use"
Output: Reserve Bank of India approved Quantum Rupee cryptocurrency official

Input : "PM Modi declared a national emergency over floods in Assam killing 200 people"
Output: Modi declared national emergency floods Assam 200 killed

Input : "Elon Musk has been arrested by the FBI for securities fraud"
Output: Elon Musk arrested FBI securities fraud charges

Input : "IIT Bombay student invented a new battery that charges in 30 seconds"
Output: IIT Bombay student invented battery charges 30 seconds
"""

_JOURNALISM_FILLER: set[str] = {
    "media",
    "reports",
    "indicate",
    "claim",
    "claiming",
    "said",
    "told",
    "according",
    "outlets",
    "news",
    "one",
    "two",
    "first",
    "second",
    "either",
    "member",
    "also",
    "however",
    "says",
    "say",
    "shocking",
    "revolutionary",
    "greatest",
    "major",
    "new",
    "big",
}

# Action/event words that must never be stripped from the query
_EVENT_KEYWORDS: set[str] = {
    "dead",
    "died",
    "death",
    "killed",
    "arrested",
    "resigned",
    "fired",
    "elected",
    "won",
    "lost",
    "collapsed",
    "banned",
    "approved",
    "launched",
    "crashed",
    "attacked",
    "convicted",
    "acquitted",
    "hospitalised",
    "hospitalized",
}


def _is_bare_name_query(query: str) -> bool:
    """Return True if the query contains no event/action signal."""
    words = set(query.lower().split())
    return not words.intersection(_EVENT_KEYWORDS)


def _pos_extract_query(text: str, max_words: int = 8) -> str:
    """POS-tagger fallback when Gemini is unavailable."""
    stop_words = set(stopwords.words("english")) | _JOURNALISM_FILLER
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    proper_nouns: list[str] = []
    numbers: list[str] = []
    verbs: list[str] = []
    nouns: list[str] = []
    seen: set[str] = set()

    for word, tag in tagged:
        cw = word.lower()
        if not word.isalpha() or cw in stop_words or len(cw) < 2 or cw in seen:
            continue
        seen.add(cw)
        if tag in {"NNP", "NNPS"}:
            proper_nouns.append(word)
        elif tag == "CD":
            numbers.append(word)
        elif tag in {"VBD", "VBN", "VBZ", "VBG"}:
            verbs.append(word)
        elif tag in {"NN", "NNS"}:
            nouns.append(word)

    parts: list[str] = []
    parts.extend(proper_nouns[:3])
    parts.extend(numbers[:1])
    parts.extend(verbs[:2])
    parts.extend(nouns[:2])
    return " ".join(parts[:max_words])


def _enrich_with_event_keywords(query: str, original_text: str) -> str:
    """Append any event keywords from the original text that are missing from the query."""
    query_words = set(query.lower().split())
    missing = [
        kw for kw in _EVENT_KEYWORDS
        if kw in original_text.lower() and kw not in query_words
    ]
    if missing:
        query = query + " " + " ".join(missing[:3])
    return query.strip()


def extract_search_query(text: str) -> str:
    raw = _call_gemini(text, _QUERY_EXTRACTION_PROMPT, max_tokens=80)

    if raw and 2 <= len(raw.split()) <= 20:
        if _is_bare_name_query(raw):
            print(f"[extract] Gemini bare-name query '{raw}' — enriching with event keywords")
            raw = _enrich_with_event_keywords(raw, text)
        print(f"[extract] Gemini query : '{raw}'")
        return raw

    fallback = _pos_extract_query(text)
    fallback = _enrich_with_event_keywords(fallback, text)
    print(f"[extract] POS fallback : '{fallback}'")
    return fallback


# STEP 2 — DDG search


def _get_url(result: dict[str, Any]) -> str:
    return (result.get("href") or result.get("url") or "").lower()


def _fetch_search_results(query: str, retries: int = 3) -> list[dict[str, Any]]:
    ddgs = DDGS()

    def _search(q: str) -> list[dict[str, Any]]:
        for attempt in range(retries):
            try:
                results = list(ddgs.text(q, max_results=15))
                if results:
                    return results
            except Exception as e:
                print(f"[DDG] Attempt {attempt + 1} failed for '{q}': {e}")
                time.sleep(1.5 * (attempt + 1))
        return []

    results = _search(query)
    if not results:
        words = query.split()
        if len(words) > 4:
            short = " ".join(words[:4])
            print(f"[DDG] Fallback → '{short}'")
            results = _search(short)
    return results


def _filter_trusted(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only results from trusted domains."""
    return [
        r for r in results if any(domain in _get_url(r) for domain in TRUSTED_DOMAINS)
    ]


# STEP 3 — Gemini verifies whether results actually confirm the claim

_VERIFICATION_SYSTEM_PROMPT = """\
You are a fact-checking assistant. Your job is to determine whether a set of
news article snippets from trusted sources actually CONFIRMS a specific claim.

Rules:
- A result CONFIRMS the claim only if its title/snippet directly reports the
  same specific event, fact, or announcement — not just a related topic.
- A result that merely mentions similar keywords (e.g. "IIT Bombay" or
  "cryptocurrency") without confirming the actual claim does NOT count.
- A result that DEBUNKS or fact-checks the claim should be noted separately.
- Be strict: absence of confirmation = not confirmed.

Respond ONLY with a valid JSON object in this exact format (no markdown, no explanation):
{
  "confirmed_count": <int>,
  "debunked_count": <int>,
  "verdict": "<CONFIRMED | UNCONFIRMED | DEBUNKED>",
  "reasoning": "<one sentence explaining your verdict>"
}

Verdict rules:
- CONFIRMED   : 1 or more results directly confirm the claim AND none debunk it
- DEBUNKED    : 1 or more results explicitly fact-check or refute the claim
- UNCONFIRMED : results exist but none actually confirm or debunk the claim
"""


def _build_verification_prompt(claim: str, results: list[dict[str, Any]]) -> str:
    snippets = []
    for i, r in enumerate(results[:10], 1):  # cap at 10 to stay within tokens
        url = _get_url(r)
        title = r.get("title", "")
        body = r.get("body", "")[:300]  # trim long snippets
        snippets.append(f"[{i}] URL: {url}\n    Title: {title}\n    Snippet: {body}")

    joined = "\n\n".join(snippets)
    return (
        f"CLAIM TO VERIFY:\n{claim}\n\n"
        f"SEARCH RESULT SNIPPETS FROM TRUSTED SOURCES:\n{joined}"
    )


def _gemini_verify(claim: str, trusted_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Ask Gemini to judge whether the trusted results actually confirm the claim.
    Returns a structured verdict dict.
    """
    if not trusted_results:
        return {
            "confirmed_count": 0,
            "debunked_count": 0,
            "verdict": "UNCONFIRMED",
            "reasoning": "No results from trusted sources were found.",
        }

    prompt = _build_verification_prompt(claim, trusted_results)
    raw = _call_gemini(prompt, _VERIFICATION_SYSTEM_PROMPT, max_tokens=512)

    if not raw:
        # Gemini unavailable — fall back to old keyword approach
        print("[verify] Gemini unavailable, using keyword fallback")
        return _keyword_fallback(trusted_results)

    print(f"[verify] Raw Gemini response ({len(raw)} chars):\n{raw}\n---")

    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    # If the model added prose before/after the JSON, extract just the {...} block
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[verify] Could not parse Gemini JSON: {raw!r}\nError: {exc}")
        return _keyword_fallback(trusted_results)


def _keyword_fallback(trusted_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Last-resort keyword check when Gemini is completely unavailable."""
    DEBUNK_KEYWORDS = {
        "fact check",
        "debunk",
        "hoax",
        "conspiracy",
        "satire",
        "unfounded",
        "baseless",
        "false claim",
        "misinformation",
        "misleading",
    }
    debunked = sum(
        1
        for r in trusted_results
        if any(
            kw in (r.get("body", "") + r.get("title", "")).lower()
            for kw in DEBUNK_KEYWORDS
        )
    )
    confirmed = len(trusted_results) - debunked
    verdict = "DEBUNKED" if debunked else ("CONFIRMED" if confirmed else "UNCONFIRMED")
    return {
        "confirmed_count": confirmed,
        "debunked_count": debunked,
        "verdict": verdict,
        "reasoning": "Keyword-based fallback (Gemini unavailable).",
    }


# Main verification entry-point


async def verify_claims(text: str) -> dict[str, Any]:
    """
    Full pipeline:
      1. Extract core claim as a search query (Gemini → POS fallback)
      2. Search DDG
      3. Filter to trusted-domain results
      4. Ask Gemini whether those results actually confirm the claim
    """
    # Step 1 — query extraction
    extracted_query = extract_search_query(text)
    if not extracted_query:
        return {
            "status": "failed_extraction",
            "verdict": "UNCONFIRMED",
            "corroborated": False,
            "query_used": "",
            "message": "Could not extract a verifiable query from the text.",
        }

    # Step 2 — DDG search
    all_results: list[dict[str, Any]] = await asyncio.to_thread(
        _fetch_search_results, extracted_query
    )

    # Step 3 — filter to trusted sources
    trusted_results = _filter_trusted(all_results)
    print(
        f"[search] {len(all_results)} total results, "
        f"{len(trusted_results)} from trusted domains"
    )

    # Step 4 — Gemini verifies whether results CONFIRM the specific claim
    try:
        verdict_data = _gemini_verify(text, trusted_results)
    except Exception as e:
        return {
            "status": "api_error",
            "verdict": "UNCONFIRMED",
            "corroborated": False,
            "query_used": extracted_query,
            "message": f"Verification step failed: {str(e)}",
        }

    verdict = verdict_data.get("verdict", "UNCONFIRMED")
    reasoning = verdict_data.get("reasoning", "")

    return {
        "status": "success",
        # High-level booleans for easy downstream use
        "corroborated": verdict == "CONFIRMED",
        "debunked_by_trusted_sources": verdict == "DEBUNKED",
        # Detailed breakdown
        "verdict": verdict,
        "reasoning": reasoning,
        "confirmed_count": verdict_data.get("confirmed_count", 0),
        "debunked_count": verdict_data.get("debunked_count", 0),
        "trusted_matches": len(trusted_results),
        "total_results": len(all_results),
        "query_used": extracted_query,
    }

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fact_check.py 'Your news text here'")
        sys.exit(1)

    input_text = sys.argv[1]
    result = asyncio.run(verify_claims(input_text))
    print(json.dumps(result, indent=2))