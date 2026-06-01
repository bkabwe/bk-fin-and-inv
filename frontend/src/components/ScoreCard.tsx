interface Props {
  score: number
}

const bands = [
  { min: 80, className: 'bg-green-100 text-green-800', label: '🟢 STRONG BUY' },
  { min: 65, className: 'bg-blue-100 text-blue-800', label: '🔵 BUY' },
  { min: 50, className: 'bg-yellow-100 text-yellow-800', label: '🟡 TAKE SMALL POSITION' },
  { min: 35, className: 'bg-orange-100 text-orange-800', label: '🟠 MONITOR' },
  { min: 20, className: 'bg-red-100 text-red-800', label: '🔴 DO NOT BUY' },
  { min: 0, className: 'bg-gray-900 text-white', label: '⛔ AVOID' }
]

export default function ScoreCard({ score }: Props) {
  const band = bands.find((b) => score >= b.min) || bands[bands.length - 1]
  return (
    <div className={`rounded-lg p-4 ${band.className}`}>
      <div className="text-xs">Score</div>
      <div className="text-2xl font-bold">{score}</div>
      <div className="text-sm font-medium">{band.label}</div>
    </div>
  )
}
