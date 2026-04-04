from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="ContraFalsum AI API",
    description="Fake News Detection and Credibility Analysis AI Service",
    version="1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health", tags=["Infrastructure"])
async def health_check():
    return {
        "status": "ok",
        "service": "ContraFalsum AI Engine",
        "message": "Ready to analyze credibility",
    }
