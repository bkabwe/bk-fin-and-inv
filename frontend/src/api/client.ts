import axios from 'axios'

const API_TIMEOUT_MS = 20000

const http = axios.create({
  baseURL: '/api',
  timeout: API_TIMEOUT_MS
})

export interface ScreenerRequest {
  universe: string
  min_score: number
  max_results: number
  custom_tickers: string[]
}

export interface ProfitRequest {
  universes: string[]
  horizon: string
  min_upside_pct: number
  max_results: number
}

export interface HoldingRequest {
  ticker: string
  shares: number
  avg_cost: number
  date_purchased: string
  notes: string
}

export interface JobResponse {
  job_id: string
  status: string
}

export interface ProgressResponse {
  job_id: string
  status: string
  screened: number
  total: number
  current_ticker: string | null
  results: Record<string, unknown>[]
  qualified: number
}

const api = {
  getHealth: async () => (await http.get('/health')).data,

  startScreener: async (payload: ScreenerRequest): Promise<JobResponse> => (await http.post('/screener/start', payload)).data,
  getScreenerProgress: async (jobId: string): Promise<ProgressResponse> => (await http.get(`/screener/${jobId}/progress`)).data,
  stopScreener: async (jobId: string): Promise<{ status: string }> => (await http.post(`/screener/${jobId}/stop`)).data,

  startProfit: async (payload: ProfitRequest): Promise<JobResponse> => (await http.post('/profit/start', payload)).data,
  getProfitProgress: async (jobId: string): Promise<ProgressResponse> => (await http.get(`/profit/${jobId}/progress`)).data,
  stopProfit: async (jobId: string): Promise<{ status: string }> => (await http.post(`/profit/${jobId}/stop`)).data,

  getPortfolio: async (): Promise<Record<string, unknown>[]> => (await http.get('/portfolio')).data,
  addHolding: async (payload: HoldingRequest) => (await http.post('/portfolio/holding', payload)).data,
  updateHolding: async (ticker: string, payload: HoldingRequest) => (await http.put(`/portfolio/holding/${ticker}`, payload)).data,
  removeHolding: async (ticker: string) => (await http.delete(`/portfolio/holding/${ticker}`)).data,
  startPortfolioAnalysis: async (): Promise<JobResponse> => (await http.post('/portfolio/analyze/start')).data,
  getPortfolioAnalysisProgress: async (jobId: string): Promise<ProgressResponse> =>
    (await http.get(`/portfolio/analyze/${jobId}/progress`)).data,

  getWatchlist: async (): Promise<Record<string, unknown>[]> => (await http.get('/watchlist')).data,
  addWatchlist: async (ticker: string) => (await http.post(`/watchlist/${ticker}`)).data,
  removeWatchlist: async (ticker: string) => (await http.delete(`/watchlist/${ticker}`)).data,
  moveWatchlistToPortfolio: async (ticker: string) => (await http.post(`/watchlist/${ticker}/portfolio`)).data,

  getAnalysis: async (ticker: string, period = '1y'): Promise<Record<string, unknown>> =>
    (await http.get(`/analysis/${ticker}`, { params: { period } })).data,
  getPriceSeries: async (ticker: string, period = '1y', interval = '1d'): Promise<Record<string, unknown>[]> =>
    (await http.get(`/analysis/${ticker}/price`, { params: { period, interval } })).data,

  getSentiment: async (ticker: string): Promise<Record<string, unknown>> => (await http.get(`/sentiment/${ticker}`)).data,
  getMarketSentiment: async (): Promise<Record<string, unknown>> => (await http.get('/sentiment/market')).data,

  getMarketOverview: async (): Promise<Record<string, { price: number; daily_pct: number }>> => (await http.get('/market/overview')).data,
  getNotifications: async (): Promise<Record<string, unknown>[]> => (await http.get('/notifications')).data,
  markNotificationsRead: async (): Promise<{ status: string }> => (await http.post('/notifications/read')).data
}

export default api
