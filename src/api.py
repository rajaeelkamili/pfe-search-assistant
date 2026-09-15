"""
API légère pour exposer le matching (partie testable sans LLM).
Lancer : uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel
from matching import load_profile, analyze_match

app = FastAPI(title="PFE Search Assistant — Matching API")
profile = load_profile()


class JobPosting(BaseModel):
    description: str


@app.post("/analyze")
def analyze(job: JobPosting):
    """Analyse une offre et retourne le score de matching + compétences/lacunes."""
    return analyze_match(job.description, profile)


@app.get("/profile")
def get_profile():
    """Retourne le profil structuré utilisé pour le matching."""
    return profile
