import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import api from '../api/client'
import ProgressBar from '../components/ProgressBar'
import StopButton from '../components/StopButton'

const allUniverses = ['S&P 500', 'NASDAQ', 'NYSE American', 'OTC']

export default function ProfitOpportunities() {
  const [universes, setUniverses] = useState<string[]>(['S&P 500', 'NASDAQ'])
  const [horizon, setHorizon] = useState('Short-Term (1–4 weeks)')
  const [minUpside, setMinUpside] = useState(15)
  const [maxResults, setMaxResults] = useState(25)
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState<string | null>(null)

  const isRunning = status === 'running'
  const { data } = useQuery({
    queryKey: ['profit-progress', jobId],
    queryFn: () => api.getProfitProgress(jobId!),
    enabled: !!jobId,
    refetchInterval: isRunning ? 1000 : false
  })

  if (data && data.status !== status) setStatus(data.status)

  const rows = ((data?.results ?? []) as Record<string, unknown>[]).slice()
  const sorted = useMemo(
    () => rows.sort((a, b) => Number(b['Projected Upside %'] ?? 0) - Number(a['Projected Upside %'] ?? 0)),
    [data?.results]
  )

  const run = async () => {
    setError(null)
    try {
      const res = await api.startProfit({ universes, horizon, min_upside_pct: minUpside, max_results: maxResults })
      setJobId(res.job_id)
      setStatus('running')
    } catch {
      setError('Unable to start profit scan. Please verify API connectivity.')
    }
  }

  const exportCsv = () => {
    const header = ['Ticker', 'Company', 'Index', 'Score', 'Current Price', 'Target Price', 'Projected Upside %', 'Est. Target Date', 'Confidence']
    const csv = [header.join(','), ...sorted.map((r) => header.map((h) => JSON.stringify(r[h] ?? '')).join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'profit-opportunities.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const top = sorted[0]

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Profit Opportunities</h1>
      {error && <div className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div>}
      <div className="grid grid-cols-1 gap-3 rounded-lg border bg-white p-4 md:grid-cols-2">
        <div className="text-sm">Universes
          <div className="mt-1 space-x-2">
            {allUniverses.map((u) => (
              <label key={u} className="inline-flex items-center gap-1 text-xs">
                <input
                  type="checkbox"
                  checked={universes.includes(u)}
                  onChange={() => setUniverses((prev) => (prev.includes(u) ? prev.filter((x) => x !== u) : [...prev, u]))}
                />
                {u}
              </label>
            ))}
          </div>
        </div>
        <div className="text-sm">Time Horizon
          {['Short-Term (1–4 weeks)', 'Medium-Term (1–6 months)', 'Long-Term (6+ months)'].map((h) => (
            <label key={h} className="ml-2 inline-flex items-center gap-1 text-xs">
              <input type="radio" checked={horizon === h} onChange={() => setHorizon(h)} />
              {h}
            </label>
          ))}
        </div>
        <label className="text-sm">Min Upside % {minUpside}
          <input type="range" min={0} max={100} value={minUpside} onChange={(e) => setMinUpside(Number(e.target.value))} className="w-full" />
        </label>
        <label className="text-sm">Max Results {maxResults}
          <input type="range" min={5} max={500} value={maxResults} onChange={(e) => setMaxResults(Number(e.target.value))} className="w-full" />
        </label>
        <button disabled={isRunning} onClick={run} className="rounded bg-green-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-60">
          Run Analysis
        </button>
      </div>

      {jobId && data && (
        <>
          <ProgressBar screened={data.screened} total={data.total} currentTicker={data.current_ticker} status={data.status} />
          {isRunning && <StopButton jobId={jobId} onStopped={() => setStatus('stopped')} kind="profit" />}
        </>
      )}

      {(status === 'stopped' || status === 'complete') && (
        <div className="space-y-3 rounded-lg border bg-white p-4">
          {top && <div className="rounded bg-green-50 p-3 text-sm">Top pick: {String(top['Ticker'])} ({String(top['Projected Upside %'])}%)</div>}
          <button onClick={exportCsv} className="rounded border px-3 py-1 text-sm">Export to CSV</button>
          <div className="h-[320px]">
            <Plot
              data={[
                {
                  type: 'bar',
                  x: sorted.map((r) => String(r['Ticker'] ?? '')),
                  y: sorted.map((r) => Number(r['Projected Upside %'] ?? 0)),
                  marker: {
                    color: sorted.map((r) => {
                      const c = String(r['Confidence'] ?? '').toLowerCase()
                      if (c.includes('full')) return '#16a34a'
                      if (c.includes('limited')) return '#f59e0b'
                      return '#3b82f6'
                    })
                  }
                }
              ]}
              layout={{ title: { text: 'Projected Upside %' }, autosize: true }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
            />
          </div>
          <div className="overflow-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr>
                  {['Ticker', 'Company', 'Index', 'Score', 'Current Price', 'Target Price', 'Projected Upside %', 'Est. Target Date', 'Confidence'].map((c) => (
                    <th key={c} className="border-b px-2 py-1 text-left">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((row, i) => (
                  <tr key={i}>
                    {['Ticker', 'Company', 'Index', 'Score', 'Current Price', 'Target Price', 'Projected Upside %', 'Est. Target Date', 'Confidence'].map((k) => (
                      <td key={k} className="border-b px-2 py-1">{String(row[k] ?? '')}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
