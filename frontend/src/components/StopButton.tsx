import { useState } from 'react'
import { Square } from 'lucide-react'
import api from '../api/client'

interface Props {
  jobId: string
  onStopped: () => void
  kind?: 'screener' | 'profit'
}

export default function StopButton({ jobId, onStopped, kind = 'screener' }: Props) {
  const [loading, setLoading] = useState(false)

  const stop = async () => {
    setLoading(true)
    try {
      if (kind === 'profit') {
        await api.stopProfit(jobId)
      } else {
        await api.stopScreener(jobId)
      }
      onStopped()
    } finally {
      setLoading(false)
    }
  }

  return (
    <button
      onClick={stop}
      disabled={loading}
      className="inline-flex items-center gap-2 rounded bg-red-600 px-3 py-2 text-sm font-semibold text-white disabled:opacity-60"
    >
      <Square size={14} />
      {loading ? 'Stopping...' : 'Stop'}
    </button>
  )
}
