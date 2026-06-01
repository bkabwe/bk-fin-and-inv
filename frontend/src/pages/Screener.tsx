import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api, { ProgressResponse } from '../api/client'
import ProgressBar from '../components/ProgressBar'
import StopButton from '../components/StopButton'

const scoreColor = (score: number) => {
  if (score >= 80) return 'text-green-700'
  if (score >= 65) return 'text-blue-700'
  if (score >= 50) return 'text-yellow-700'
  if (score >= 35) return 'text-orange-700'
  return 'text-red-700'
}

export default function Screener() {
  const [universe, setUniverse] = useState('sp500')
  const [minScore, setMinScore] = useState(60)
  const [maxResults, setMaxResults] = useState(25)
  const [custom, setCustom] = useState('')
  const [horizons, setHorizons] = useState<string[]>([])
  const [sector, setSector] = useState('')
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState('idle')
  const [sortKey, setSortKey] = useState('Score')
  const [sortAsc, setSortAsc] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const isRunning = status === 'running'

  const progress = useQuery<ProgressResponse>({
    queryKey: ['screener-progress', jobId],
    queryFn: () => api.getScreenerProgress(jobId!),
    enabled: !!jobId,
    refetchInterval: isRunning ? 1000 : false
  })

  const data = progress.data
  if (data && data.status !== status) setStatus(data.status)

  const run = async () => {
    setError(null)
    try {
      const res = await api.startScreener({
        universe,
        min_score: minScore,
        max_results: maxResults,
        custom_tickers: custom.split(/[\s,]+/).filter(Boolean)
      })
      setJobId(res.job_id)
      setStatus('running')
    } catch {
      setError('Unable to start screener. Please check API service.')
    }
  }

  const rows = useMemo(() => {
    const list = (data?.results ?? []) as Record<string, unknown>[]
    const sorted = [...list].sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      if (typeof av === 'number' && typeof bv === 'number') return sortAsc ? av - bv : bv - av
      return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av))
    })
    return sorted
  }, [data?.results, sortAsc, sortKey])

  const exportCsv = () => {
    const header = ['Ticker', 'Company', 'Score', 'Recommendation', 'Time Horizon', 'Current Price', 'Entry Price', 'Target Price']
    const csv = [header.join(','), ...rows.map((r) => header.map((h) => JSON.stringify(r[h] ?? '')).join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'screener-results.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const summary = data
    ? status === 'stopped'
      ? `⏹ Stopped at ${data.screened} stocks`
      : `✅ Scanned ${data.screened} stocks → ${data.qualified} scored ≥${minScore} → Showing top ${Math.min(maxResults, rows.length)} by score`
    : ''

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Stock Screener</h1>
      {error && <div className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div>}
      <div className="grid grid-cols-1 gap-3 rounded-lg border bg-white p-4 md:grid-cols-2">
        <label className="text-sm">Universe
          <select className="mt-1 w-full rounded border p-2" value={universe} onChange={(e) => setUniverse(e.target.value)}>
            <option value="sp500">S&P 500</option>
            <option value="nasdaq">NASDAQ</option>
            <option value="nyseamerican">NYSE American</option>
            <option value="otc">OTC</option>
            <option value="custom">Custom</option>
          </select>
        </label>
        <label className="text-sm">Min Score {minScore}
          <input type="range" min={0} max={100} value={minScore} onChange={(e) => setMinScore(Number(e.target.value))} className="w-full" />
        </label>
        <label className="text-sm">Max Results {maxResults}
          <input type="range" min={5} max={500} value={maxResults} onChange={(e) => setMaxResults(Number(e.target.value))} className="w-full" />
        </label>
        <label className="text-sm">Sector Filter
          <input className="mt-1 w-full rounded border p-2" value={sector} onChange={(e) => setSector(e.target.value)} placeholder="e.g. Technology" />
        </label>
        <div className="text-sm">Time Horizon
          {['Short-Term', 'Medium-Term', 'Long-Term'].map((h) => (
            <label key={h} className="ml-2 inline-flex items-center gap-1 text-xs">
              <input type="checkbox" checked={horizons.includes(h)} onChange={() => setHorizons((p) => (p.includes(h) ? p.filter((x) => x !== h) : [...p, h]))} />
              {h}
            </label>
          ))}
        </div>
        {universe === 'custom' && (
          <label className="text-sm md:col-span-2">Custom tickers
            <textarea className="mt-1 w-full rounded border p-2" rows={3} value={custom} onChange={(e) => setCustom(e.target.value)} />
          </label>
        )}
        <button disabled={isRunning} onClick={run} className="rounded bg-green-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-60">
          Run Screener
        </button>
      </div>

      {jobId && data && (
        <>
          <ProgressBar screened={data.screened} total={data.total} currentTicker={data.current_ticker} status={data.status} />
          {isRunning && (
            <div className="flex items-center gap-4">
              <StopButton jobId={jobId} onStopped={() => setStatus('stopped')} />
              <span className="text-sm text-gray-500">Qualified so far: {data.qualified}</span>
            </div>
          )}
        </>
      )}

      {(status === 'stopped' || status === 'complete') && data && (
        <div className="space-y-2 rounded-lg border bg-white p-4">
          <div className="text-sm font-medium">{summary}</div>
          <button onClick={exportCsv} className="rounded border px-3 py-1 text-sm">Export to CSV</button>
          <div className="overflow-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr>
                  {['Ticker', 'Company', 'Score', 'Recommendation', 'Time Horizon', 'Current Price', 'Entry Price', 'Target Price'].map((col) => (
                    <th
                      key={col}
                      className="cursor-pointer border-b px-2 py-1 text-left"
                      onClick={() => {
                        if (sortKey === col) setSortAsc(!sortAsc)
                        else {
                          setSortKey(col)
                          setSortAsc(false)
                        }
                      }}
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={i}>
                    <td className="border-b px-2 py-1">{String(row['Ticker'] ?? '')}</td>
                    <td className="border-b px-2 py-1">{String(row['Company'] ?? '')}</td>
                    <td className={`border-b px-2 py-1 font-semibold ${scoreColor(Number(row['Score'] ?? 0))}`}>{String(row['Score'] ?? '')}</td>
                    <td className="border-b px-2 py-1">{String(row['Recommendation'] ?? '')}</td>
                    <td className="border-b px-2 py-1">{String(row['Time Horizon'] ?? '')}</td>
                    <td className="border-b px-2 py-1">{String(row['Current Price'] ?? '')}</td>
                    <td className="border-b px-2 py-1">{String(row['Entry Price'] ?? '')}</td>
                    <td className="border-b px-2 py-1">{String(row['Target Price'] ?? '')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-xs text-gray-600">Score legend: 80–100 🟢, 65–79 🔵, 50–64 🟡, 35–49 🟠, 20–34 🔴, 0–19 ⛔</div>
        </div>
      )}
    </div>
  )
}
