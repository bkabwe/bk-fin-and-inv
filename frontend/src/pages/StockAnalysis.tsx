import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import api from '../api/client'
import PriceChart from '../components/PriceChart'
import ScoreCard from '../components/ScoreCard'

export default function StockAnalysis() {
  const [ticker, setTicker] = useState('AAPL')
  const [period, setPeriod] = useState('1y')

  const analysis = useQuery({ queryKey: ['analysis', ticker, period], queryFn: () => api.getAnalysis(ticker, period) })
  const prices = useQuery({ queryKey: ['analysis-price', ticker, period], queryFn: () => api.getPriceSeries(ticker, period, '1d') })

  if (analysis.error || prices.error) {
    return <div className="rounded border border-red-200 bg-red-50 p-4 text-red-700">Unable to load analysis. Please check API availability.</div>
  }

  const score = Number(analysis.data?.score ?? 0)
  const breakdown = analysis.data?.score_breakdown as Record<string, number> | undefined

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Stock Analysis</h1>
      <div className="flex gap-2">
        <input className="rounded border p-2" value={ticker} onChange={(e) => setTicker(e.target.value)} />
        <select className="rounded border p-2" value={period} onChange={(e) => setPeriod(e.target.value)}>
          {['1mo', '3mo', '6mo', '1y', '2y', '5y'].map((p) => (
            <option key={p} value={p}>{p.toUpperCase()}</option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
        <ScoreCard score={score} />
        <div className="rounded border bg-white p-4 text-sm">Recommendation: {String(analysis.data?.recommendation ?? '')}</div>
        <div className="rounded border bg-white p-4 text-sm">Entry/Target: {String(analysis.data?.entry_price ?? '')} / {String(analysis.data?.target_price ?? '')}</div>
        <div className="rounded border bg-white p-4 text-sm">Stop: {String(analysis.data?.stop_loss ?? '')}</div>
      </div>

      <PriceChart rows={prices.data ?? []} />

      <div className="h-[320px] rounded border bg-white p-2">
        <Plot
          data={[
            {
              type: 'bar',
              x: breakdown ? Object.keys(breakdown) : [],
              y: breakdown ? Object.values(breakdown) : []
            }
          ]}
          layout={{ title: { text: 'Score Breakdown' }, autosize: true }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
        />
      </div>

      <div className="rounded border bg-white p-4 text-sm">
        <strong>Tabs:</strong> Technical, Fundamental, Patterns, News & Sentiment
        <pre className="mt-2 overflow-auto text-xs">{JSON.stringify(analysis.data ?? {}, null, 2)}</pre>
      </div>
    </div>
  )
}
