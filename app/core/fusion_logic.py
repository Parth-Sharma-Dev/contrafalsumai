from typing import Any


def calculate_final_score(
    base_prediction: dict[str, Any],
    domain_data: dict[str, Any],
    fact_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Fuses three pillars into a single credibility score.

    Priority order (highest → lowest):
      Pillar C — Fact-check  (50%)
      Pillar B — Domain      (30%)
      Pillar A — Content     (20%)

    Hard overrides (applied BEFORE weighted scoring):
      • Debunked by trusted source  → score capped at 15  → FAKE
      • Domain is untrusted/absent  → score capped at 35  → FAKE
      • Fact verdict is UNCONFIRMED → score capped at 40  → FAKE
      (overrides stack: the lowest applicable cap wins)
    """

    # ── Pillar A: Content score (TF-IDF model) ────────────────────────────────
    confidence: float = base_prediction["confidence"]
    if base_prediction["label"] == 1:          # model says REAL
        content_score: float = confidence * 100
    else:                                       # model says FAKE
        content_score = (1.0 - confidence) * 100

    # ── Pillar B: Domain score ────────────────────────────────────────────────
    domain_trusted: bool = False
    domain_score:   float = 0.0               # pessimistic default (not 50!)

    if domain_data["status"] == "success":
        domain_trusted = True
        domain_score   = 100.0
        if domain_data.get("is_high_risk_tld"):
            domain_score -= 40.0
            domain_trusted = False
        if domain_data.get("high_risk_age"):
            domain_score -= 40.0
            domain_trusted = False
        domain_score = max(domain_score, 0.0)
    # status != "success" → domain_score stays 0, domain_trusted stays False

    # ── Pillar C: Fact-check score ────────────────────────────────────────────
    fact_score:   float = 0.0                  # pessimistic default (not 50!)
    fact_verdict: str   = fact_data.get("verdict", "UNCONFIRMED")
    is_debunked:  bool  = fact_data.get("debunked_by_trusted_sources", False)

    if fact_data["status"] == "success":
        if is_debunked:
            fact_score = 0.0
        elif fact_data.get("corroborated"):    # verdict == "CONFIRMED"
            fact_score = 100.0
        else:                                   # UNCONFIRMED
            fact_score = 0.0                   # treat as FAKE signal

    # ── Weighted fusion: Fact 50% | Domain 30% | Content 20% ─────────────────
    raw_score: float = (
        (fact_score    * 0.50) +
        (domain_score  * 0.30) +
        (content_score * 0.20)
    )

    # ── Hard overrides (stack — lowest cap wins) ──────────────────────────────
    score_cap: float = 100.0
    override_reasons: list[str] = []

    if is_debunked:
        score_cap = min(score_cap, 15.0)
        override_reasons.append("debunked by trusted source")

    if not domain_trusted:
        score_cap = min(score_cap, 35.0)
        override_reasons.append("domain is untrusted or unavailable")

    if fact_verdict == "UNCONFIRMED" and not is_debunked:
        score_cap = min(score_cap, 40.0)
        override_reasons.append("fact-check returned UNCONFIRMED")

    final_score: float = min(raw_score, score_cap)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict: str = "REAL" if final_score >= 50.0 else "FAKE"

    return {
        "final_verdict":   verdict,
        "credibility_score": round(final_score, 2),
        "fusion_breakdown": {
            "fact_score":    round(fact_score,    2),
            "domain_score":  round(domain_score,  2),
            "content_score": round(content_score, 2),
        },
        "override_applied":  bool(override_reasons),
        "override_reasons":  override_reasons,
        "raw_weighted_score": round(raw_score, 2),
    }