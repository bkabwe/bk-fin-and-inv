import { useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import api from '../api/client'
import ProgressBar from '../components/ProgressBar'

export default function Portfolio() {
  const qc = useQueryClient()
  const [form, setForm] = useState({ ticker: '', shares: 0, avg_cost: 0, date_purchased: '', notes: '' })
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState('idle')

  const portfolio = useQuery({ queryKey: ['portfolio'], queryFn: api.getPortfolio })
  const analysis = useQuery({
    queryKey: ['portfolio-progress', jobId],
    queryFn: () => api.getPortfolioAnalysisProgress(jobId!),
    enabled: !!jobId,
    refetchInterval: status === 'running' ? 1000 : false
  })

  if (analysis.data && analysis.data.status !== status) setStatus(analysis.data.status)

  const add = async () => {
    await api.addHolding(form)
    setForm({ ticker: '', shares: 0, avg_cost: 0, date_purchased: '', notes: '' })
    qc.invalidateQueries({ queryKey: ['portfolio'] })
  }

  const run = async () => {
    const res = await api.startPortfolioAnalysis()
    setJobId(res.job_id)
    setStatus('running')
  }

  const rows = (analysis.data?.results ?? []) as Record<string, unknown>[]
  const allocation = useMemo(() => rows.map((r) => Number(r['Current Value'] ?? 0)), [analysis.data?.results])

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Portfolio</h1>
      <div className="grid grid-cols-1 gap-2 rounded border bg-white p-4 md:grid-cols-5">
        {['ticker', 'shares', 'avg_cost', 'date_purchased', 'notes'].map((k) => (
          <input
            key={k}
            className="rounded border p-2 text-sm"
            placeholder={k}
            value={String(form[k as keyof typeof form])}
            onChange={(e) => setForm((p) => ({ ...p, [k]: k === 'ticker' || k === 'date_purchased' || k === 'notes' ? e.target.value : Number(e.target.value) }))}
          />
        ))}
        <button className="rounded bg-blue-600 px-3 py-2 text-sm text-white" onClick={add}>Add Holding</button>
      </div>

      <div className="rounded border bg-white p-4">
        <table className="min-w-full text-sm">
          <thead><tr><th className="text-left">Ticker</th><th>Shares</th><th>Avg Cost</th><th /></tr></thead>
          <tbody>
            {(portfolio.data ?? []).map((h, i) => (
              <tr key={i}>
                <td>{String(h.ticker ?? '')}</td><td>{String(h.shares ?? '')}</td><td>{String(h.avg_cost ?? '')}</td>
                <td className="space-x-2">
                  <button className="rounded border px-2 py-1" onClick={() => api.removeHolding(String(h.ticker ?? '')).then(() => qc.invalidateQueries({ queryKey: ['portfolio'] }))}>Remove</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <button className="rounded bg-green-600 px-4 py-2 text-sm text-white" onClick={run}>Analyze All Holdings</button>

      {analysis.data && <ProgressBar screened={analysis.data.screened} total={analysis.data.total} currentTicker={analysis.data.current_ticker} status={analysis.data.status} />}

      {rows.length > 0 && (
        <>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="h-[320px] rounded border bg-white p-2">
              <Plot
                data={[{ type: 'pie', labels: rows.map((r) => String(r['Ticker'] ?? '')), values: allocation }]}
                layout={{ title: { text: 'Portfolio Allocation' }, autosize: true }}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
              />
            </div>
            <div className="h-[320px] rounded border bg-white p-2">
              <Plot
                data={[{ type: 'bar', x: rows.map((r) => String(r['Ticker'] ?? '')), y: rows.map((r) => Number(r['P&L %'] ?? 0)) }]}
                layout={{ title: { text: 'P&L %' }, autosize: true }}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
              />
            </div>
          </div>
          <div className="space-y-2">
            {rows.map((r, i) => (
              <details key={i} className="rounded border bg-white p-3">
                <summary className="cursor-pointer font-medium">{String(r['Ticker'] ?? '')} — {String(r['Recommendation'] ?? '')}</summary>
                <pre className="mt-2 overflow-auto text-xs">{JSON.stringify(r['Projection'] ?? {}, null, 2)}</pre>
              </details>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
