import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import api from '../api/client'

export default function NewsSentiment() {
  const [ticker, setTicker] = useState('AAPL')
  const { data, error } = useQuery({ queryKey: ['sentiment', ticker], queryFn: () => api.getSentiment(ticker) })
  const market = useQuery({ queryKey: ['sentiment-market'], queryFn: api.getMarketSentiment })

  if (error) {
    return <div className="rounded border border-red-200 bg-red-50 p-4 text-red-700">Unable to load sentiment data.</div>
  }

  const headlines = (data?.headlines as Record<string, unknown>[] | undefined) ?? []
  const score = Number(data?.sentiment_score ?? 0)

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">News & Sentiment</h1>
      <input className="rounded border p-2" value={ticker} onChange={(e) => setTicker(e.target.value)} />
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="rounded border bg-white p-4 text-sm">Short interest ratio: {String(data?.short_ratio ?? 'N/A')}</div>
        <div className="rounded border bg-white p-4 text-sm">Put/Call ratio: {String(data?.put_call_ratio ?? 'N/A')}</div>
        <div className="rounded border bg-white p-4 text-sm">Sentiment Label: {String(data?.sentiment_label ?? 'N/A')}</div>
      </div>

      <div className="h-[260px] rounded border bg-white p-2">
        <Plot
          data={[
            {
              type: 'indicator',
              mode: 'gauge+number',
              value: score,
              gauge: { axis: { range: [-1, 1] } }
            }
          ]}
          layout={{ title: { text: 'Sentiment Gauge' }, autosize: true }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
        />
      </div>

      <div className="h-[260px] rounded border bg-white p-2">
        <Plot
          data={[{ type: 'scatter', mode: 'lines+markers', x: headlines.map((_, i) => i + 1), y: headlines.map((h) => Number(h.score ?? 0)) }]}
          layout={{ title: { text: 'Sentiment Trend' }, autosize: true }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
        />
      </div>

      <div className="rounded border bg-white p-4">
        <h3 className="mb-2 text-sm font-semibold">Headlines</h3>
        <table className="min-w-full text-sm">
          <thead><tr><th>Headline</th><th>Label</th><th>Score</th></tr></thead>
          <tbody>
            {headlines.map((h, i) => (
              <tr key={i}><td>{String(h.headline ?? '')}</td><td>{String(h.label ?? '')}</td><td>{String(h.score ?? '')}</td></tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="rounded border bg-white p-4 text-sm">
        <h3 className="mb-2 font-semibold">Market-wide sentiment</h3>
        <pre className="overflow-auto text-xs">{JSON.stringify(market.data ?? {}, null, 2)}</pre>
      </div>
    </div>
  )
}
