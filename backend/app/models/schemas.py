from pydantic import BaseModel
from typing import List, Optional

class AnalyzeRequest(BaseModel):
    polygon: List[List[float]]

class AnalyzeResponse(BaseModel):
    area_acres: float
    road_distance_ft: float
    power_distance_mi: Optional[float] = None
    flood_overlap_pct: float
    slope_mean_pct: float
    risk_score: float