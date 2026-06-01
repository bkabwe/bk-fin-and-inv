import { useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

export default function Watchlist() {
  const qc = useQueryClient()
  const [ticker, setTicker] = useState('')
  const [sort, setSort] = useState<'Score' | 'Ticker' | 'Price'>('Score')

  const { data = [] } = useQuery({ queryKey: ['watchlist'], queryFn: api.getWatchlist })

  const rows = useMemo(() => {
    const copy = [...data]
    copy.sort((a, b) => {
      if (sort === 'Ticker') return String(a.Ticker ?? '').localeCompare(String(b.Ticker ?? ''))
      return Number(b[sort] ?? 0) - Number(a[sort] ?? 0)
    })
    return copy
  }, [data, sort])

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Watchlist</h1>
      <div className="flex gap-2">
        <input className="rounded border p-2" value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="Ticker" />
        <button
          className="rounded bg-blue-600 px-3 py-2 text-sm text-white"
          onClick={() => api.addWatchlist(ticker).then(() => qc.invalidateQueries({ queryKey: ['watchlist'] }))}
        >
          Add
        </button>
      </div>
      <div className="space-x-2 text-sm">
        Sort:
        <button className="rounded border px-2 py-1" onClick={() => setSort('Score')}>Score</button>
        <button className="rounded border px-2 py-1" onClick={() => setSort('Ticker')}>Alpha</button>
        <button className="rounded border px-2 py-1" onClick={() => setSort('Price')}>Price</button>
      </div>
      <div className="rounded border bg-white p-4">
        <table className="min-w-full text-sm">
          <thead><tr><th>Ticker</th><th>Price</th><th>Score</th><th>Recommendation</th><th /></tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td>{String(r.Ticker ?? '')}</td>
                <td>{String(r.Price ?? '')}</td>
                <td>{String(r.Score ?? '')}</td>
                <td>{String(r.Recommendation ?? '')}</td>
                <td className="space-x-2">
                  <button className="rounded border px-2 py-1" onClick={() => api.removeWatchlist(String(r.Ticker ?? '')).then(() => qc.invalidateQueries({ queryKey: ['watchlist'] }))}>Remove</button>
                  <button className="rounded border px-2 py-1" onClick={() => api.moveWatchlistToPortfolio(String(r.Ticker ?? '')).then(() => qc.invalidateQueries({ queryKey: ['watchlist'] }))}>Move</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
