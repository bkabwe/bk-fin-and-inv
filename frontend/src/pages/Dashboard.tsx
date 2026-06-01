import { useQuery } from '@tanstack/react-query'
import api from '../api/client'
import Notification from '../components/Notification'

export default function Dashboard() {
  const { data: market = {}, error } = useQuery({ queryKey: ['market-overview'], queryFn: api.getMarketOverview })
  const { data: notifications = [] } = useQuery({
    queryKey: ['notifications-dashboard'],
    queryFn: api.getNotifications,
    refetchInterval: 30000
  })

  if (error) {
    return <div className="rounded border border-red-200 bg-red-50 p-4 text-red-700">API unreachable. Please ensure backend is running.</div>
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Market Overview</h1>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
        {['SPY', 'QQQ', 'DIA', 'IWM'].map((ticker) => {
          const item = market[ticker]
          return (
            <div key={ticker} className="rounded-lg border bg-white p-4">
              <div className="text-sm text-gray-500">{ticker}</div>
              <div className="text-xl font-semibold">${item?.price ?? '--'}</div>
              <div className={`text-sm ${Number(item?.daily_pct ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>{item?.daily_pct ?? '--'}%</div>
            </div>
          )
        })}
      </div>
      <Notification notifications={notifications} />
    </div>
  )
}
