import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.analyze import router as analyze_router
from app.api.mississippi_leads import router as mississippi_leads_router
from app.api.state_leads import router as state_leads_router
from app.services.state_service_registry import configured_state_codes
from app.settings import ALLOWED_CORS_ORIGINS, GZIP_MINIMUM_SIZE, state_runtime_file_diagnostics

logger = logging.getLogger("state-runtime")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=GZIP_MINIMUM_SIZE)


@app.on_event("startup")
def log_runtime_file_diagnostics():
    for state_code in configured_state_codes():
        for name, info in state_runtime_file_diagnostics(state_code).items():
            logger.info(
                "State runtime file state=%s name=%s cwd=%s project_root=%s resolved_path=%s exists=%s size_bytes=%s",
                state_code,
                name,
                info["cwd"],
                info["project_root"],
                info["path"],
                info["exists"],
                info["size_bytes"],
            )

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok", "service": "landintel-backend", "health": "/health"}

app.include_router(analyze_router)
app.include_router(state_leads_router)
app.include_router(mississippi_leads_router)
