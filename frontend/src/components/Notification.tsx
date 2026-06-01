interface Props {
  notifications: Record<string, unknown>[]
}

export default function Notification({ notifications }: Props) {
  return (
    <div className="rounded-lg border bg-white p-4">
      <h3 className="mb-2 text-sm font-semibold">Recent Notifications</h3>
      <div className="space-y-2">
        {notifications.slice(0, 10).map((note, idx) => (
          <div key={idx} className="rounded border p-2 text-xs">
            <div className="font-semibold">{String(note.title ?? note.type ?? 'Info')}</div>
            <div>{String(note.message ?? '')}</div>
            <div className="text-gray-500">{String(note.timestamp ?? '')}</div>
          </div>
        ))}
        {!notifications.length && <div className="text-xs text-gray-500">No notifications.</div>}
      </div>
    </div>
  )
}
