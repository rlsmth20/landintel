from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.analysis_engine import analyze_land
from app.db import get_db

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, db: Session = Depends(get_db)):

    result = analyze_land(request.polygon, db)

    return result