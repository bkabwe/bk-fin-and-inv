#!/bin/bash
set -e

echo "🔴 Starting Redis..."
brew services start redis 2>/dev/null || echo "Redis already running"

echo "⚙️  Starting Celery worker..."
celery -A api.worker worker --loglevel=info --concurrency=4 &

echo "🚀 Starting FastAPI backend..."
uvicorn api.main:app --reload --port 8000 &

echo "⚛️  Starting React frontend..."
cd frontend && npm run dev &

echo ""
echo "✅ All services started!"
echo "   Frontend:  http://localhost:5173"
echo "   API:       http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo "   Streamlit: streamlit run app.py  (run separately)"
echo ""
echo "Press Ctrl+C to stop all services"
wait
