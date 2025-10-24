# Local Quickstart

The quickest way to run Mini-XDR locally is through the project start script. This guide assumes
macOS or Linux with Python 3.10+ and Node.js 20 installed.

## 1. Clone and prerequisites

```bash
# inside your workspace
git clone <repo-url>
cd mini-xdr
python3 --version  # expect 3.10 or newer
node --version     # expect v20.x
```

The helper script `scripts/start-all.sh` checks ports and prerequisites, but you can prepare the
runtime manually for clarity.

**Note**: This project uses automated documentation enforcement. When you make code changes that are confirmed working, the system will validate that corresponding documentation updates are included in your commits. See [`docs-enforcement.md`](docs-enforcement.md) for details.

## 2. Backend environment

```bash
cp backend/env.example backend/.env
# IMPORTANT: Edit backend/.env and set at minimum:
# - API_KEY (generate a strong 64+ character secret)
# - JWT_SECRET_KEY (generate a strong random secret for JWT signing)
# - ENCRYPTION_KEY (generate a strong random secret for data encryption)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

Key environment variables are described in
[`environment-config.md`](environment-config.md); the defaults use SQLite storage at
`backend/xdr.db`. For local development, you must set `API_KEY` and `JWT_SECRET_KEY` in
`backend/.env` before starting the backend.

## 3. Frontend environment

```bash
cd frontend
npm install
cd ..
```

The Next.js app expects `NEXT_PUBLIC_API_BASE` and `NEXT_PUBLIC_API_KEY`. For local testing, create
`frontend/.env.local` and set these values (use the same API_KEY from backend/.env):

```
NEXT_PUBLIC_API_BASE=http://localhost:8000
NEXT_PUBLIC_API_KEY=<same-value-as-API_KEY-in-backend-.env>
```

## 4. Start services

Either run everything with the orchestrator:

```bash
./scripts/start-all.sh
```

Or start each service manually in separate terminals:

```bash
# Terminal 1 - FastAPI
source .venv/bin/activate
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Next.js UI
cd frontend
npm run dev
```

Once running:

- API docs: http://localhost:8000/docs (requires API key headers for secured endpoints).
- Frontend: http://localhost:3000 (registration/login page).
- SOC dashboard: http://localhost:3000/incidents (after login).
- Agent orchestration: http://localhost:3000/agents (component reads from `/api/agents/orchestrate`).

## 5. Seed data (optional)

Use the signed request helper to simulate inbound events:

```bash
python3 scripts/auth/send_signed_request.py \
  --base-url http://localhost:8000 \
  --path /ingest/multi \
  --method POST \
  --body @scripts/samples/cowrie-event.json
```

Confirm new incidents appear in the dashboard and that `/api/incidents` returns data.

## 6. Shut down

Stop uvicorn and Next.js with `Ctrl+C`. If you used `start-all.sh`, the script kills running
processes automatically on exit.
