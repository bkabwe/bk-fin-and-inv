from modules.env import load_environment

load_environment()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.analysis import router as analysis_router
from api.routes.market import router as market_router
from api.routes.portfolio import router as portfolio_router
from api.routes.profit import router as profit_router
from api.routes.screener import router as screener_router
from api.routes.sentiment import router as sentiment_router
from api.routes.track_record import router as track_record_router
from api.routes.watchlist import router as watchlist_router

app = FastAPI(title="BK Fin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market_router, prefix="/api")
app.include_router(screener_router, prefix="/api")
app.include_router(profit_router, prefix="/api")
app.include_router(portfolio_router, prefix="/api")
app.include_router(watchlist_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(sentiment_router, prefix="/api")
app.include_router(track_record_router, prefix="/api")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
