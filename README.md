# ContraFalsum AI Service

**The AI Service backend for the ContraFalsum project** — a sophisticated fake news detection and credibility analysis system powered by hybrid intelligence: ML-based content analysis, source verification, and real-time fact-checking.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [System Methodology](#system-methodology)
4. [API Endpoints](#api-endpoints)
5. [Installation & Setup](#installation--setup)
6. [Configuration](#configuration)

---

## Overview

ContraFalsum is a **three-pillar hybrid system** that detects fake news through:

- **Pillar A (20%)**: Stylometric analysis — ML-based content classification with explainability
- **Pillar B (30%)**: Source verification — Domain trustworthiness via TLD and age analysis
- **Pillar C (50%)**: Fact-checking — Real-time claim verification against trusted news sources

These pillars run in parallel and fuse results using weighted scoring with intelligent override rules to produce a final credibility verdict and score.

---

## Architecture

```
User Input (Text + Optional URL)
        ↓
   ┌────────────────────────────────────────┐
   │    PILLAR A: STYLOMETRIC ANALYSIS      │
   │         (ML-based Content)             │
   │  • TF-IDF vectorization                │
   │  • Trained fake news classifier        │
   │  • LIME explainability weights         │
   │  • Runs synchronously                  │
   └────────────────────────────────────────┘
        ↓
   ┌─────────────────────────────────────────────────────┐
   │              PARALLEL EXECUTION                      │
   ├──────────────────────┬──────────────────────────────┤
   │                      │                              │
   │ PILLAR B:            │ PILLAR C:                    │
   │ SOURCE VERIFY        │ FACT-CHECKING               │
   │ (Domain Trust)       │ (Claim Verification)        │
   │ • TLD risk check     │ • Query extraction (Gemini) │
   │ • Domain age via     │ • DuckDuckGo search         │
   │   WHOIS             │ • Filter to trusted sources │
   │ • Runs async        │ • Gemini verification       │
   └──────────────────────┴──────────────────────────────┘
        ↓
   ┌────────────────────────────────────────┐
   │   FUSION LOGIC                         │
   │   • Weighted score: Fact (50%)         │
   │              Domain (30%)              │
   │              Content (20%)             │
   │   • Hard overrides applied             │
   └────────────────────────────────────────┘
        ↓
   Final Verdict + Credibility Score + Explainability
```

---

## System Methodology

### Phase 1: Request Handling

**Endpoint:** `POST /analyze`

**Input:**

- `text` (string, ≥10 characters): the news content to verify
- `url` (optional): source URL

**Processing Flow:**

1. Validate input (minimum length check)
2. Branch into three parallel analysis streams
3. Synchronously execute Pillar A (ML-based)
4. Launch async tasks for Pillars B & C (network I/O)
5. Wait for all tasks to complete
6. Fuse results and return

---

### Pillar A: Stylometric Analysis

**Weight: 20% of final score | Execution: Synchronous**

#### Text Preprocessing

Input text undergoes aggressive normalization:

1. **Lowercase conversion** — `"Hello WORLD"` → `"hello world"`
2. **HTML removal** — Strip any `<tags>`
3. **URL removal** — Remove URLs and emails
4. **Special character removal** — Keep only alphanumeric + spaces
5. **Tokenization** — Split into words using NLTK
6. **Stop word removal** — Filter common words (but preserve negations like "not", "no")
7. **Lemmatization** — Convert to root form (e.g., "running" → "run", "studies" → "study")

**Example:**

```
Input:  "Check out this shocking news! Visit www.fake.com"
Output: "check shocking news"
```

#### TF-IDF Classification

- **Pipeline used:** Pre-trained scikit-learn pipeline stored in `ml_assets/fake_news_pipeline.pkl`
- **Model type:** TF-IDF vectorizer + trained classifier
- **Output:**
  - `label`: 0 (FAKE) or 1 (REAL)
  - `confidence`: probability [0.0 - 1.0]
  - `verdict`: "FAKE" or "REAL"

**Score Calculation:**

```
If model says REAL (label=1):
  content_score = confidence × 100

If model says FAKE (label=0):
  content_score = (1.0 - confidence) × 100

Range: 0–100
```

#### LIME Explainability

- **Purpose:** Identify which words contributed to the "fake" prediction
- **Method:** LIME (Local Interpretable Model-agnostic Explanations)
- **Output:** Dictionary of suspicious words + their weight

Example output:

```json
{
  "shocking": 0.2345,
  "unbelievable": 0.1876,
  "exclusive": 0.1543
}
```

---

### Pillar B: Source Domain Verification

**Weight: 30% of final score | Execution: Async (parallel to Pillar C)**

#### URL Parsing

- Extract root domain from full URL
- Remove `www.` prefix
- Normalize to lowercase
- Example: `https://www.example.com/news/article` → `example.com`

#### TLD Risk Assessment

- Check against hardcoded **high-risk TLDs**:
  - `.xyz`, `.top`, `.click`, `.news`, `.biz`, `.info`
- **Penalty:** If high-risk TLD found → `-40 points` from domain score

#### Domain Age Verification (via WHOIS)

- Query WHOIS database asynchronously for domain creation date
- Calculate age in days: `today - creation_date`
- **Risk threshold:** Domains < 180 days old are flagged
- **Penalty:** If brand new → `-40 points`

#### Domain Score Calculation

```
Base domain_score = 100.0

If high_risk_tld: domain_score -= 40
If high_risk_age (< 6 months): domain_score -= 40

Final: domain_score = max(domain_score, 0.0)
```

**Output:**

```json
{
  "status": "success",
  "domain": "example.com",
  "is_high_risk_tld": false,
  "domain_age_days": 1250,
  "high_risk_age": false
}
```

---

### Pillar C: Fact-Checking

**Weight: 50% of final score | Execution: Async (parallel to Pillar B)**

This pillar verifies whether the news claim is corroborated by trusted sources.

#### Query Extraction (Gemini + POS Fallback)

**Goal:** Convert free-form text into a concise, searchable query

**Method 1 - Gemini AI (Primary):**

- Use Gemini 2.5 Flash model with system prompt
- Extract **one** core factual claim (4-9 words)
- Include named entities, people, organizations, locations
- Exclude filler words and subjective adjectives
- Implement rate-limit retry logic (automatic backoff for 429 errors)

**Example:**

```
Input:  "Scientists at MIT have cured cancer using a new AI drug called NeuroHeal"
Output: "MIT scientists cured cancer NeuroHeal AI drug"
```

**Method 2 - POS Fallback (if Gemini unavailable):**

- Use NLTK part-of-speech tagging
- Extract proper nouns (NNP), verbs (VBD/VBN), numbers (CD)
- Rank by POS priority and frequency
- Assemble into 8-word query max

#### DuckDuckGo Search

- Search using extracted query
- Fetch up to 15 results
- Implement retry logic with exponential backoff
- If results sparse, fallback to first 4 words of query

#### Trusted Domain Filtering

- Filter results to **only** trusted news sources
- Curated list of 100+ trusted domains including:
  - Major news: Reuters, AP News, BBC, NYT, Guardian, etc.
  - Fact-check sites: Snopes, FactCheck.org, PolitiFact, etc.
  - Regional & international: AlJazeera, DW, France24, etc.
  - Country-specific: Indian Express, Dawn, etc.

**Result:** Returns only results from trusted sources

#### Gemini Verification

- **Input:** Original claim + search result snippets from trusted sources
- **Task:** Gemini judges if snippets actually CONFIRM the claim
- **Logic:**
  - ✅ **CONFIRMED** — Results directly report the same event/fact
  - ❌ **DEBUNKED** — Results fact-check or refute the claim
  - ⚠️ **UNCONFIRMED** — Results exist but don't confirm/debunk

**Output:**

```json
{
  "confirmed_count": 2,
  "debunked_count": 0,
  "verdict": "CONFIRMED",
  "reasoning": "2 reputable sources confirm the claim"
}
```

#### Fallback (Keyword-based)

- If Gemini unavailable, scan snippets for debunk keywords:
  - "fact check", "debunk", "hoax", "conspiracy", "satire", "false claim", etc.
- Count occurrences to determine verdict

---

### Phase 2: Fusion Logic

Combines all three pillars with sophisticated override rules.

#### Normalize Individual Scores

```
Pillar A (Content):    0–100 (derived from ML confidence)
Pillar B (Domain):     0–100 (health check or 0 if unavailable)
Pillar C (Fact):       0–100 (0 if debunked, 100 if confirmed, 0 if unconfirmed)
```

#### Weighted Average

```
Raw Score = (Fact × 0.50) + (Domain × 0.30) + (Content × 0.20)
Range: 0–100
```

**Example:**

```
Fact:    85 (confirmed by trusted sources)
Domain:  40 (high-risk TLD)
Content: 65 (ML model confidence)

Raw = (85 × 0.50) + (40 × 0.30) + (65 × 0.20)
    = 42.5 + 12 + 13
    = 67.5
```

#### Hard Overrides (Stacking Logic)

**Override Priority (highest to lowest):**

1. **DEBUNKED OVERRIDE** (Highest Priority)
   - If fact-check returns DEBUNKED
   - Cap score at **15.0 max**
   - Reason: Strong evidence of misinformation

2. **CONFIRMED OVERRIDE** (Higher Priority)
   - If fact-check returns CONFIRMED
   - Set floor at **85.0 minimum**
   - Ignores domain penalties
   - Reason: Direct corroboration overrides distrust in domain

3. **DOMAIN PENALTY** (Standard)
   - If domain untrusted: cap at **35.0 max**

4. **UNCONFIRMED PENALTY** (Standard)
   - If fact-check unconfirmed: cap at **40.0 max**

**Stacking Example:**

```
Scenario: Article from sketchy domain, fact-check says CONFIRMED
Raw score: 50
Domain untrusted → would cap at 35
BUT confirmed override → set floor at 85
Final: max(50, 85) = 85
```

#### Final Score Calculation

```
Final Score = min(max(raw_score, floor), cap)
```

#### Convert to Verdict

```
If final_score >= 50.0:  verdict = "REAL"
If final_score < 50.0:   verdict = "FAKE"
```

---

### Phase 3: Response Construction

**Return Structure:**

```json
{
  "final_verdict": "REAL" | "FAKE",
  "credibility_score": 0–100,
  "explainability": {
    "suspicious_words": {
      "shocking": 0.234,
      "alleged": 0.187
    }
  },
  "fusion_breakdown": {
    "fact_score": 85.0,
    "domain_score": 40.0,
    "content_score": 65.0
  },
  "raw_data": {
    "source_verification": { ... },
    "fact_checking": { ... }
  }
}
```

---

## API Endpoints

### Full Analysis

```
POST /analyze
```

**Request:**

```json
{
  "text": "The string of news content to analyze",
  "url": "https://source.com/article (optional)"
}
```

**Response:**

```json
{
  "final_verdict": "REAL",
  "credibility_score": 78.5,
  "explainability": { "suspicious_words": {...} },
  "fusion_breakdown": { ... },
  "raw_data": { ... }
}
```

### Stylometric Analysis Only

```
POST /analyze/stylometric
```

**Request:**

```json
{
  "text": "The string of news content to analyze"
}
```

### Source/Domain Analysis Only

```
POST /analyze/source
```

**Request:**

```json
{
  "url": "https://example.com"
}
```

### Fact-Check Only

```
POST /analyze/fact-check
```

**Request:**

```json
{
  "text": "The string of news content to fact-check"
}
```

### Health Check

```
GET /health
```

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- GEMINI_API_KEY environment variable set
- Docker (optional)

### Local Setup

1. **Clone the repository:**

```bash
git clone <repo-url>
cd FakeNewsAIService
```

2. **Create virtual environment:**

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set environment variables:**

```bash
# Create .env file in root directory
GEMINI_API_KEY=your_api_key_here
```

5. **Run the API:**

```bash
uvicorn app.main:app --reload
```

API will be available at `http://localhost:8000`

### Docker Setup

```bash
docker-compose up --build
```

---

## Configuration

### Environment Variables

| Variable         | Required | Description                           |
| ---------------- | -------- | ------------------------------------- |
| `GEMINI_API_KEY` | Yes      | Google Gemini API key for LLM queries |

### System Tuning

**Trusted Domains List:** Edit `app/services/fact_check.py` — `TRUSTED_DOMAINS` list
**High-Risk TLDs:** Edit `app/services/source_verify.py` — `HIGH_RISK_TLDS` set
**Domain Age Threshold:** Edit `app/services/source_verify.py` (currently 180 days)
**ML Pipeline:** Replace `ml_assets/fake_news_pipeline.pkl` with new trained model

---

## Key Design Decisions

| Feature                     | Why                                        |
| --------------------------- | ------------------------------------------ |
| **Fact-check weighted 50%** | Direct verification is most reliable       |
| **Async Pillars B & C**     | Network I/O doesn't block computation      |
| **LIME explainability**     | Users see WHY text was flagged             |
| **Override stacking**       | Confirmed claims override domain distrust  |
| **Trusted domain curation** | Prevents false positives from fringe sites |
| **Rate-limit retry logic**  | Handles Gemini API quota gracefully        |
| **Debunk hard cap (15.0)**  | Prevents credible-looking disinformation   |

---

## Architecture Files

```
app/
  ├── main.py                  # FastAPI application setup
  ├── api/
  │   └── routes.py            # API endpoint definitions
  ├── core/
  │   └── fusion_logic.py       # Score fusion & override logic
  ├── services/
  │   ├── fact_check.py        # Pillar C: Fact-checking
  │   ├── source_verify.py     # Pillar B: Domain verification
  │   └── stylometric.py       # Pillar A: ML classification
  └── utils/
      └── text_processing.py   # Text normalization pipeline
ml_assets/
  └── fake_news_pipeline.pkl   # Trained TF-IDF + classifier model
```

---

## License

See LICENSE file for details.
