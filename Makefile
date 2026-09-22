.PHONY: setup-backend test-backend run-api setup-frontend run-frontend build-frontend

setup-backend:
	cd backend && python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,pqc]"

test-backend:
	cd backend && python -m pytest

run-api:
	cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

setup-frontend:
	npm install --prefix frontend

run-frontend:
	npm --prefix frontend run dev

build-frontend:
	npm --prefix frontend run build
