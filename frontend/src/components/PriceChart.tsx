import Plot from 'react-plotly.js'

interface Props {
  rows: Record<string, unknown>[]
}

export default function PriceChart({ rows }: Props) {
  if (!rows.length) return <div className="rounded border bg-white p-4 text-sm text-gray-500">No price data available.</div>

  const dates = rows.map((r) => String(r.Date ?? r.Datetime ?? ''))
  const open = rows.map((r) => Number(r.Open ?? 0))
  const high = rows.map((r) => Number(r.High ?? 0))
  const low = rows.map((r) => Number(r.Low ?? 0))
  const close = rows.map((r) => Number(r.Close ?? 0))

  return (
    <div className="h-[420px] rounded border bg-white p-2">
      <Plot
        data={[{ type: 'candlestick', x: dates, open, high, low, close }]}
        layout={{ title: { text: 'Price Chart' }, autosize: true }}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}
