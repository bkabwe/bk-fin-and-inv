import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Portfolio from './pages/Portfolio'
import Watchlist from './pages/Watchlist'
import Screener from './pages/Screener'
import StockAnalysis from './pages/StockAnalysis'
import NewsSentiment from './pages/NewsSentiment'
import ProfitOpportunities from './pages/ProfitOpportunities'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="/watchlist" element={<Watchlist />} />
        <Route path="/screener" element={<Screener />} />
        <Route path="/analysis" element={<StockAnalysis />} />
        <Route path="/sentiment" element={<NewsSentiment />} />
        <Route path="/profit" element={<ProfitOpportunities />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}
