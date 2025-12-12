import { useParams } from 'react-router-dom'
import { useDiscovery, useDiscoveryMetrics, useDiscoveryCycles, useStopDiscovery } from '@/hooks/useDiscovery'
import { useWebSocket } from '@/hooks/useWebSocket'
import Card from '@/components/common/Card'
import Loading from '@/components/common/Loading'
import CostChart from '@/components/dashboard/CostChart'
import CycleTimeline from '@/components/dashboard/CycleTimeline'
import TaskBreakdown from '@/components/dashboard/TaskBreakdown'
import { Activity, DollarSign, FileSearch, FlaskConical, StopCircle } from 'lucide-react'

// Transform metrics data for CostChart
function transformCostData(metrics: any) {
  if (!metrics?.cost_per_cycle || metrics.cost_per_cycle.length === 0) {
    return []
  }
  let cumulative = 0
  return metrics.cost_per_cycle.map((cost: number, index: number) => {
    cumulative += cost
    return {
      cycle: index + 1,
      cost: cost,
      cumulative: cumulative
    }
  })
}

// Transform metrics data for CycleTimeline
function transformCycleData(metrics: any) {
  if (!metrics?.cost_per_cycle || metrics.cost_per_cycle.length === 0) {
    return []
  }
  return metrics.cost_per_cycle.map((_: number, index: number) => ({
    cycle_number: index + 1,
    findings_added: metrics.findings_per_cycle?.[index] || 0,
    tasks_completed: Math.round((metrics.tasks_completed || 0) / metrics.cost_per_cycle.length)
  }))
}

// Event configuration for activity feed display
function getEventConfig(type: string): { icon: string; label: string; bgColor: string; textColor: string } {
  const configs: Record<string, { icon: string; label: string; bgColor: string; textColor: string }> = {
    'cycle_started': {
      icon: '🔄',
      label: 'Cycle Started',
      bgColor: 'bg-blue-50 dark:bg-blue-900/30 border-blue-200 dark:border-blue-800',
      textColor: 'text-blue-700 dark:text-blue-300'
    },
    'cycle_completed': {
      icon: '✓',
      label: 'Cycle Complete',
      bgColor: 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-800',
      textColor: 'text-green-700 dark:text-green-300'
    },
    'task_started': {
      icon: '▶',
      label: 'Task Started',
      bgColor: 'bg-yellow-50 dark:bg-yellow-900/30 border-yellow-200 dark:border-yellow-800',
      textColor: 'text-yellow-700 dark:text-yellow-300'
    },
    'task_completed': {
      icon: '✓',
      label: 'Task Complete',
      bgColor: 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-800',
      textColor: 'text-green-700 dark:text-green-300'
    },
    'task_failed': {
      icon: '✗',
      label: 'Task Failed',
      bgColor: 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-800',
      textColor: 'text-red-700 dark:text-red-300'
    },
    'progress_update': {
      icon: 'ℹ',
      label: 'Progress',
      bgColor: 'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600',
      textColor: 'text-gray-700 dark:text-gray-300'
    },
    'budget_warning': {
      icon: '⚠',
      label: 'Budget Warning',
      bgColor: 'bg-orange-50 dark:bg-orange-900/30 border-orange-200 dark:border-orange-800',
      textColor: 'text-orange-700 dark:text-orange-300'
    },
    'discovery_started': {
      icon: '🚀',
      label: 'Discovery Started',
      bgColor: 'bg-purple-50 dark:bg-purple-900/30 border-purple-200 dark:border-purple-800',
      textColor: 'text-purple-700 dark:text-purple-300'
    },
    'discovery_completed': {
      icon: '🎉',
      label: 'Discovery Complete',
      bgColor: 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-800',
      textColor: 'text-green-700 dark:text-green-300'
    },
    'discovery_failed': {
      icon: '❌',
      label: 'Discovery Failed',
      bgColor: 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-800',
      textColor: 'text-red-700 dark:text-red-300'
    },
  }
  return configs[type] || {
    icon: '•',
    label: type.replace(/_/g, ' '),
    bgColor: 'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600',
    textColor: 'text-gray-700 dark:text-gray-300'
  }
}

export default function Dashboard() {
  const { discoveryId } = useParams<{ discoveryId: string }>()
  const { data: status, isLoading } = useDiscovery(discoveryId!)
  const { data: metrics } = useDiscoveryMetrics(discoveryId!)
  const { data: cyclesData } = useDiscoveryCycles(discoveryId!)
  const { messages, isConnected } = useWebSocket(discoveryId!)
  const stopDiscovery = useStopDiscovery()

  // Extract all tasks from cycles for task distribution chart
  const allTasks = cyclesData?.flatMap(cycle =>
    cycle.tasks.map(task => ({
      ...task,
      cycle_number: cycle.cycle_number
    }))
  ) || []

  if (isLoading) return <Loading message="Loading discovery..." />
  if (!status) return <div>Discovery not found</div>

  const handleStop = async () => {
    if (confirm('Are you sure you want to stop this discovery?')) {
      await stopDiscovery.mutateAsync(discoveryId!)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Discovery Dashboard
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            ID: {discoveryId}
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            isConnected
              ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
              : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
          }`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
          {status.status === 'running' && (
            <button
              onClick={handleStop}
              className="flex items-center space-x-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg"
            >
              <StopCircle className="w-4 h-4" />
              <span>Stop Discovery</span>
            </button>
          )}
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard
          title="Total Cost"
          value={`$${status.total_cost?.toFixed(2) || '0.00'}`}
          icon={<DollarSign className="w-6 h-6" />}
          color="blue"
        />
        <MetricCard
          title="Current Cycle"
          value={status.current_cycle || 0}
          icon={<Activity className="w-6 h-6" />}
          color="purple"
        />
        <MetricCard
          title="Findings"
          value={status.findings_count || 0}
          icon={<FileSearch className="w-6 h-6" />}
          color="green"
        />
        <MetricCard
          title="Hypotheses"
          value={status.hypotheses_count || 0}
          icon={<FlaskConical className="w-6 h-6" />}
          color="orange"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CostChart data={transformCostData(metrics)} />
        <CycleTimeline cycles={transformCycleData(metrics)} />
      </div>

      <TaskBreakdown tasks={allTasks} />

      {/* Live Feed */}
      <Card title="Live Activity Feed" subtitle="Real-time updates from the discovery process">
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {messages.length === 0 ? (
            <p className="text-gray-500 dark:text-gray-400 text-center py-8">
              Waiting for events...
            </p>
          ) : (
            messages.slice().reverse().map((msg, idx) => {
              const config = getEventConfig(msg.type)
              return (
                <div
                  key={idx}
                  className={`p-3 rounded-lg border ${config.bgColor}`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start gap-2 flex-1 min-w-0">
                      <span className="text-lg flex-shrink-0">{config.icon}</span>
                      <div className="min-w-0 flex-1">
                        <div className={`font-medium text-sm ${config.textColor}`}>
                          {config.label}
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-300 mt-0.5 truncate">
                          {msg.data?.message || msg.type}
                        </div>
                      </div>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 flex-shrink-0">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              )
            })
          )}
        </div>
      </Card>
    </div>
  )
}

interface MetricCardProps {
  title: string
  value: string | number
  icon: React.ReactNode
  color: 'blue' | 'purple' | 'green' | 'orange'
}

function MetricCard({ title, value, icon, color }: MetricCardProps) {
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-200',
    purple: 'bg-purple-100 text-purple-600 dark:bg-purple-900 dark:text-purple-200',
    green: 'bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-200',
    orange: 'bg-orange-100 text-orange-600 dark:bg-orange-900 dark:text-orange-200',
  }

  return (
    <Card>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-2">{value}</p>
        </div>
        <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
          {icon}
        </div>
      </div>
    </Card>
  )
}
