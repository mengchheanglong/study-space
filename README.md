# Studyspace

Studyspace is a local-first learning workspace that combines:
- `web-app`: Next.js dashboard/UI for study workflows
- `services/local-rag-ai-assistant`: FastAPI + LangChain + Ollama RAG backend
- `services/transcript-whisper`: FastAPI Whisper transcription service

## Project Status

- Transcript workflow: usable
- Study RAG workflow: usable
- Docs workflow: usable
- IDE workflow: usable for local experimentation
- Forum workflow: **unfinished / experimental**
  The forum is intended to become a Discord-like study channel with AI chat support, but it is not complete and behavior may change.

## Architecture

- Frontend/UI: `web-app` (Next.js 16 + React + TypeScript)
- Backends:
  - Transcript API on `127.0.0.1:8000`
  - Local RAG API on `127.0.0.1:9999`
- Optional local model runtime: Ollama (`11434`)
- IDE codebot model: Qwen (`qwen2.5-coder:3b` by default)

## Repository Layout

- `start-studyspace.ps1`: local stack launcher
- `studyspace.code-workspace`: VS Code workspace
- `web-app/`: main web app
- `services/`: backend services
- `services/local-rag-ai-assistant/my_docs/`: local document input folder (kept empty in git)
- `services/local-rag-ai-assistant/study_collections/`: collection metadata + generated artifacts (runtime data)

## Prerequisites

- Node.js 20.9+
- Python 3.10+ (3.13 recommended)
- `ffmpeg` on PATH for transcript extraction
- Ollama installed locally for RAG/AI-assisted features

## Quick Start

1. Install Web App dependencies:
```powershell
cd web-app
npm install
cd ..
```

2. Dry-run launcher:
```powershell
npm run dev:dry-run
```

3. Start full stack:
```powershell
npm run dev
```

4. Open:
- Web App: `http://localhost:3000/dashboard`
- Transcript API docs: `http://127.0.0.1:8000/docs`
- Local RAG API health: `http://127.0.0.1:9999/health`

## Quick Run (Recommended)
### Option A: One command from terminal

```powershell
npm run dev
```

This runs `start-studyspace.ps1`, prepares local env files, and launches the Web App + both backend services.

### Option B: VS Code workspace flow

1. Open `studyspace.code-workspace` in VS Code.
2. Run task: `Start Studyspace` (from `.vscode/tasks.json`).
3. Use the workspace with preconfigured folders and local dev defaults.

## Verification

Web App checks:
```powershell
cd web-app
npm run lint
npm run typecheck
```

Python syntax checks:
```powershell
python -m compileall -q services/transcript-whisper/src
python -m compileall -q services/local-rag-ai-assistant
```

## Environment Notes

- `start-studyspace.ps1` writes `web-app/.env.local` during runtime bootstrap.
- IDE codebot defaults to Qwen via `LOCAL_CODE_ASSISTANT_MODEL=qwen2.5-coder:3b`.
- RAG model overrides:
  - `OLLAMA_EMBED_MODEL`
  - `OLLAMA_LLM_MODEL`
- Transcript service overrides are documented in `services/transcript-whisper/.env.example`.

## Known Limitations

- Forum feature is unfinished and currently positioned as an experimental preview.
- Full backend test execution requires installing Python dev dependencies (`pytest`, etc.).
- Some build/test operations may fail in restricted environments with strict process/network policies.
- On Python 3.14+, some LangChain dependencies may print Pydantic v1 compatibility warnings. Launcher import checks ignore warnings and only fail on real import errors.
