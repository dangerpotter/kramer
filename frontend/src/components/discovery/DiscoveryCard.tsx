import { useNavigate } from 'react-router-dom'
import { Clock, DollarSign, RefreshCw, Lightbulb, FileText, Beaker } from 'lucide-react'
import { cn } from '@/utils/cn'
import { DiscoveryStatus } from '@/types/discovery'

interface DiscoverySummary {
  discovery_id: string
  objective: string
  model: string
  status: string
  current_cycle: number
  total_cost: number
  created_at: string
  started_at?: string
  completed_at?: string
  findings_count?: number
  hypotheses_count?: number
  papers_count?: number
}

interface DiscoveryCardProps {
  discovery: DiscoverySummary
}

const statusConfig: Record<string, { label: string; className: string }> = {
  [DiscoveryStatus.RUNNING]: {
    label: 'Running',
    className: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
  },
  [DiscoveryStatus.COMPLETED]: {
    label: 'Completed',
    className: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
  },
  [DiscoveryStatus.FAILED]: {
    label: 'Failed',
    className: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300',
  },
  [DiscoveryStatus.STOPPED]: {
    label: 'Stopped',
    className: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
  },
  [DiscoveryStatus.PENDING]: {
    label: 'Pending',
    className: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300',
  },
}

function formatDate(dateString: string): string {
  const date = new Date(dateString)
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatDuration(startedAt?: string, completedAt?: string): string | null {
  if (!startedAt) return null

  const start = new Date(startedAt)
  const end = completedAt ? new Date(completedAt) : new Date()
  const durationMs = end.getTime() - start.getTime()

  const hours = Math.floor(durationMs / (1000 * 60 * 60))
  const minutes = Math.floor((durationMs % (1000 * 60 * 60)) / (1000 * 60))

  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }
  return `${minutes}m`
}

export default function DiscoveryCard({ discovery }: DiscoveryCardProps) {
  const navigate = useNavigate()
  const status = statusConfig[discovery.status] || statusConfig[DiscoveryStatus.PENDING]
  const duration = formatDuration(discovery.started_at, discovery.completed_at)

  const handleClick = () => {
    navigate(`/dashboard/${discovery.discovery_id}`)
  }

  return (
    <div
      onClick={handleClick}
      className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5 hover:border-primary-500 dark:hover:border-primary-500 hover:shadow-md transition-all cursor-pointer"
    >
      {/* Header with objective and status */}
      <div className="flex items-start justify-between gap-4 mb-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white line-clamp-2 flex-1">
          {discovery.objective}
        </h3>
        <span
          className={cn(
            'px-2.5 py-1 text-xs font-medium rounded-full whitespace-nowrap',
            status.className
          )}
        >
          {status.label}
        </span>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <RefreshCw className="w-4 h-4" />
          <span>{discovery.current_cycle} cycles</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <DollarSign className="w-4 h-4" />
          <span>${discovery.total_cost.toFixed(2)}</span>
        </div>
        {discovery.findings_count !== undefined && (
          <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
            <Lightbulb className="w-4 h-4" />
            <span>{discovery.findings_count} findings</span>
          </div>
        )}
        {discovery.hypotheses_count !== undefined && (
          <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
            <Beaker className="w-4 h-4" />
            <span>{discovery.hypotheses_count} hypotheses</span>
          </div>
        )}
      </div>

      {/* Footer with dates */}
      <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-500 pt-3 border-t border-gray-100 dark:border-gray-700">
        <div className="flex items-center gap-1">
          <Clock className="w-3.5 h-3.5" />
          <span>Created {formatDate(discovery.created_at)}</span>
        </div>
        {duration && (
          <span className="text-gray-400 dark:text-gray-500">
            Duration: {duration}
          </span>
        )}
      </div>
    </div>
  )
}
