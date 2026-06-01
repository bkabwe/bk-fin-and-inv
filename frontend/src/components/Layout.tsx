import { Bell } from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

interface Props {
  children: React.ReactNode
}

const links = [
  ['/', 'Dashboard'],
  ['/portfolio', 'Portfolio'],
  ['/watchlist', 'Watchlist'],
  ['/screener', 'Screener'],
  ['/analysis', 'Stock Analysis'],
  ['/sentiment', 'News & Sentiment'],
  ['/profit', 'Profit Opportunities']
]

export default function Layout({ children }: Props) {
  const location = useLocation()
  const { data: notifications = [] } = useQuery({
    queryKey: ['notifications-layout'],
    queryFn: api.getNotifications,
    refetchInterval: 30000
  })

  const unread = notifications.filter((n) => !Boolean(n.read)).length

  return (
    <div className="flex min-h-screen">
      <aside className="w-64 border-r bg-white p-4">
        <div className="mb-4 text-lg font-bold">📈 BK Finance</div>
        <nav className="space-y-1">
          {links.map(([to, label]) => {
            const active = location.pathname === to
            return (
              <Link key={to} to={to} className={`block rounded px-3 py-2 text-sm ${active ? 'bg-blue-100 text-blue-800' : 'hover:bg-slate-100'}`}>
                {label}
              </Link>
            )
          })}
        </nav>
      </aside>
      <main className="flex-1 p-6">
        <div className="mb-4 flex justify-end">
          <button className="relative rounded border bg-white p-2">
            <Bell size={18} />
            {unread > 0 && (
              <span className="absolute -right-1 -top-1 rounded-full bg-red-600 px-1.5 text-[10px] text-white">{unread}</span>
            )}
          </button>
        </div>
        {children}
      </main>
    </div>
  )
}
