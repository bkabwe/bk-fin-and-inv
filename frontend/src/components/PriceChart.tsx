import Plot from 'react-plotly.js'

interface Props {
  rows: Record<string, unknown>[]
}

export default function PriceChart({ rows }: Props) {
  if (!rows.length) return <div className="rounded border bg-white p-4 text-sm text-gray-500">No price data available.</div>

  // Filter out data points with any missing/null OHLC value to prevent
  // false collapse-to-zero artifacts on the chart.
  const valid = rows.filter(
    (r) =>
      r.Open != null && r.High != null && r.Low != null && r.Close != null &&
      !isNaN(Number(r.Open)) && !isNaN(Number(r.High)) && !isNaN(Number(r.Low)) && !isNaN(Number(r.Close))
  )

  if (!valid.length) return <div className="rounded border bg-white p-4 text-sm text-gray-500">No price data available.</div>

  const dates = valid.map((r) => String(r.Date ?? r.Datetime ?? ''))
  const open = valid.map((r) => Number(r.Open))
  const high = valid.map((r) => Number(r.High))
  const low = valid.map((r) => Number(r.Low))
  const close = valid.map((r) => Number(r.Close))

  return (
    <div className="h-[420px] rounded border bg-white p-2">
      <Plot
        data={[{ type: 'candlestick', x: dates, open, high, low, close }]}
        layout={{ title: { text: 'Price Chart (split-adjusted)' }, autosize: true }}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}
