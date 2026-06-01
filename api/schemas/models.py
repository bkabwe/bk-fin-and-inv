from __future__ import annotations

from pydantic import BaseModel, Field


class ScreenerRequest(BaseModel):
    universe: str = "sp500"
    min_score: int = 60
    max_results: int = 25
    custom_tickers: list[str] = Field(default_factory=list)


class ProfitRequest(BaseModel):
    universes: list[str] = Field(default_factory=lambda: ["S&P 500", "NASDAQ"])
    horizon: str = "Short-Term (1–4 weeks)"
    min_upside_pct: float = 15.0
    max_results: int = 25


class HoldingRequest(BaseModel):
    ticker: str
    shares: float
    avg_cost: float
    date_purchased: str
    notes: str = ""


class JobResponse(BaseModel):
    job_id: str
    status: str


class ProgressResponse(BaseModel):
    job_id: str
    status: str
    screened: int
    total: int
    current_ticker: str | None
    results: list[dict]
    qualified: int
