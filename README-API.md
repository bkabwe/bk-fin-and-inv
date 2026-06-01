# BK Fin API + React Frontend

## Prerequisites
1. Homebrew
2. Node.js 18+
3. Python 3.11+
4. Redis

## Installation
```bash
brew install redis node
pip3 install -r requirements-api.txt
cd frontend && npm install
```

## Running
```bash
chmod +x start-api.sh && ./start-api.sh
```

## Running Streamlit instead
```bash
streamlit run app.py
```

## API docs
http://localhost:8000/docs

## Troubleshooting
- Redis not running: `brew services start redis`
- Port conflicts: free ports 5173/8000/6379
- Celery not connecting: verify Redis on localhost:6379
