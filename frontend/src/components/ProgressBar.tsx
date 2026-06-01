interface Props {
  screened: number
  total: number
  currentTicker: string | null
  status: string
}

export default function ProgressBar({ screened, total, currentTicker, status }: Props) {
  const percent = total > 0 ? Math.min(100, (screened / total) * 100) : 0

  return (
    <div className="space-y-2 rounded-lg border bg-white p-4">
      <div className="h-3 w-full rounded bg-gray-200">
        <div className="h-3 rounded bg-blue-500" style={{ width: `${percent}%` }} />
      </div>
      <div className="text-sm font-medium">
        {screened.toLocaleString()} / {total.toLocaleString()} stocks screened ({percent.toFixed(1)}%)
      </div>
      {status === 'stopped' && <p className="text-sm text-red-600">⏹ Stopped at {screened.toLocaleString()} stocks</p>}
      {status === 'complete' && <p className="text-sm text-green-600">✅ Scan complete — {screened.toLocaleString()} stocks screened</p>}
      {currentTicker && <p className="text-xs text-gray-500">Currently analysing: {currentTicker}</p>}
    </div>
  )
}
