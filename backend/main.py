from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from app.api.analyze import router as analyze_router
from app.api.mississippi_leads import router as mississippi_leads_router
from app.settings import ALLOWED_CORS_ORIGINS, GZIP_MINIMUM_SIZE

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=GZIP_MINIMUM_SIZE)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(analyze_router)
app.include_router(mississippi_leads_router)
