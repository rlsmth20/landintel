# Mississippi MVP Deployment

This deployment path assumes:

- the Mississippi product uses the read-only FastAPI backend in [backend/main.py](/C:/Users/Rainer/landrisk/backend/main.py)
- the frontend uses the Next app in [frontend](/C:/Users/Rainer/landrisk/frontend)
- product data already exists at:
  - [app_ready_mississippi_leads.parquet](/C:/Users/Rainer/landrisk/data/tax_published/ms/app_ready_mississippi_leads.parquet)
  - [mississippi_lead_explorer_meta.json](/C:/Users/Rainer/landrisk/frontend/public/data/mississippi_lead_explorer_meta.json)
  - [mississippi_lead_explorer_geometries.json](/C:/Users/Rainer/landrisk/frontend/public/data/mississippi_lead_explorer_geometries.json)

## Backend

### Required env vars

Backend env vars are documented in [backend/.env.example](/C:/Users/Rainer/landrisk/backend/.env.example).

Minimum recommended production settings:

```env
APP_HOST=0.0.0.0
APP_PORT=8000
ALLOWED_CORS_ORIGINS=https://your-frontend.example.com
MISSISSIPPI_APP_READY_PATH=/app/data/tax_published/ms/app_ready_mississippi_leads.parquet
MISSISSIPPI_META_PATH=/app/frontend/public/data/mississippi_lead_explorer_meta.json
MISSISSIPPI_GEOMETRY_PATH=/app/frontend/public/data/mississippi_lead_explorer_geometries.json
```

Optional API guardrails:

```env
LEADS_DEFAULT_LIMIT=200
LEADS_MAX_LIMIT=250
GEOMETRY_DEFAULT_LIMIT=350
GEOMETRY_MAX_LIMIT=500
GZIP_MINIMUM_SIZE=1024
```

### Local backend run

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.runtime.txt
python run.py
```

Alternative uvicorn launch:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Production gunicorn launch:

```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 2
```

The API is read-only and serves:

- `/health`
- `/api/summary`
- `/api/presets`
- `/api/leads`
- `/api/leads/{parcel_row_id}`
- `/api/leads/geometry`

### Backend notes

- CORS is controlled by `ALLOWED_CORS_ORIGINS`
- gzip compression is enabled for JSON responses
- list responses are paginated by default and capped server-side
- geometry responses are subset-based and do not return full-state geometry by default
- use [requirements.runtime.txt](/C:/Users/Rainer/landrisk/backend/requirements.runtime.txt) for deployed API runtime installs
- keep [requirements.txt](/C:/Users/Rainer/landrisk/backend/requirements.txt) only for the broader geospatial/dev stack
- the runtime requirements still include the lightweight SQL packages needed for the existing `/analyze` route import chain

## Frontend

### Required env vars

Frontend env vars are documented in [frontend/.env.local.example](/C:/Users/Rainer/landrisk/frontend/.env.local.example).

Local:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Production:

```env
NEXT_PUBLIC_API_BASE_URL=https://your-api.example.com
```

If the frontend and backend are deployed behind the same origin and reverse proxy, `NEXT_PUBLIC_API_BASE_URL` can be left empty and the app will use same-origin API paths.

### Local frontend run

```bash
cd frontend
npm install
copy .env.local.example .env.local
npm run dev
```

Production build:

```bash
npm run build
npm run start
```

## Local dev flow

1. Start backend on `http://localhost:8000`
2. Set `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`
3. Start frontend on `http://localhost:3000`
4. Verify:
   - presets load
   - filtered list loads
   - parcel detail loads
   - geometry loads for selected/visible parcels

## MVP hosting recommendation

Simple MVP deployment:

- backend: single FastAPI process behind gunicorn/uvicorn on a small VM or container host
- frontend: Vercel, Netlify, or a small Node host
- CORS: restrict to the frontend origin only
- data: keep parquet and geometry/meta artifacts on the backend host filesystem

This is acceptable for the Mississippi MVP because the API is read-only and the dataset is local-file-backed.

## Remaining production work after MVP

- add reverse-proxy caching for `/api/summary`, `/api/presets`, and low-cardinality filtered responses
- move geometry delivery from JSON support files to tile or viewport-based geometry service for larger scale
- add structured app logging and request metrics
- add deployment secrets/config management rather than checked-in example env files
