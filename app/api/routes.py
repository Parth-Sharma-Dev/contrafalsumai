import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any
from app.services.stylometric import analyze_with_lime, get_base_prediction
from app.services.source_verify import analyze_domain
from app.services.fact_check import verify_claims
from app.core.fusion_logic import calculate_final_score

router: APIRouter = APIRouter()


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10)
    url: str | None = Field(None)

class TextOnlyRequest(BaseModel):
    text: str = Field(..., min_length=10)

class UrlOnlyRequest(BaseModel):
    url: str


@router.post("/analyze", summary="Run Full Hybrid Verification", tags=["Core Intelligence"])
async def full_analysis(payload: AnalyzeRequest) -> dict[str, Any]:
    try:
        # Step 1: Run Pillar A (Synchronous ML code)
        base_ml_result = get_base_prediction(payload.text)
        lime_weights = analyze_with_lime(payload.text)
        
        # Step 2: Prepare async tasks for Pillars B & C
        domain_task = analyze_domain(payload.url) if payload.url else None
        fact_task = verify_claims(payload.text)
        
        # Step 3: Execute network-heavy tasks concurrently!
        if domain_task:
            domain_analysis, fact_check_analysis = await asyncio.gather(domain_task, fact_task)
        else:
            domain_analysis = {"status": "no_url_provided"}
            fact_check_analysis = await fact_task
            
        # Step 4: Run the Fusion Logic (Now includes Debunk Override)
        fusion_result = calculate_final_score(base_ml_result, domain_analysis, fact_check_analysis)

        # Step 5: Construct the final payload
        return {
            "final_verdict": fusion_result["final_verdict"],
            "credibility_score": fusion_result["credibility_score"],
            "explainability": {
                "suspicious_words": lime_weights
            },
            "fusion_breakdown": fusion_result["fusion_breakdown"],
            "raw_data": {
                "source_verification": domain_analysis,
                "fact_checking": fact_check_analysis
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/stylometric", tags=["Pillars"])
async def analyze_stylometric(payload: TextOnlyRequest) -> dict[str, Any]:
    return {
        "base_prediction": get_base_prediction(payload.text),
        "suspicious_words": analyze_with_lime(payload.text)
    }


@router.post("/analyze/source", tags=["Pillars"])
async def analyze_source(payload: UrlOnlyRequest) -> dict[str, Any]:
    return await analyze_domain(payload.url)


@router.post("/analyze/fact-check", tags=["Pillars"])
async def analyze_facts(payload: TextOnlyRequest) -> dict[str, Any]:
    return await verify_claims(payload.text)