import { useParams } from 'react-router-dom'
import { useDiscovery, useDiscoveryMetrics, useStopDiscovery } from '@/hooks/useDiscovery'
import { useWebSocket } from '@/hooks/useWebSocket'
import Card from '@/components/common/Card'
import Loading from '@/components/common/Loading'
import CostChart from '@/components/dashboard/CostChart'
import CycleTimeline from '@/components/dashboard/CycleTimeline'
import TaskBreakdown from '@/components/dashboard/TaskBreakdown'
import { Activity, DollarSign, FileSearch, FlaskConical, StopCircle } from 'lucide-react'

export default function Dashboard() {
  const { discoveryId } = useParams<{ discoveryId: string }>()
  const { data: status, isLoading } = useDiscovery(discoveryId!)
  const { data: metrics } = useDiscoveryMetrics(discoveryId!)
  const { messages, isConnected } = useWebSocket(discoveryId!)
  const stopDiscovery = useStopDiscovery()

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
        <CostChart data={metrics?.cost_history || []} />
        <CycleTimeline cycles={metrics?.cycles || []} />
      </div>

      <TaskBreakdown tasks={metrics?.tasks || []} />

      {/* Live Feed */}
      <Card title="Live Activity Feed" subtitle="Real-time updates from the discovery process">
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {messages.length === 0 ? (
            <p className="text-gray-500 dark:text-gray-400 text-center py-8">
              Waiting for events...
            </p>
          ) : (
            messages.slice().reverse().map((msg, idx) => (
              <div
                key={idx}
                className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="font-medium text-sm text-gray-900 dark:text-white">
                      {msg.type}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                      {JSON.stringify(msg.data, null, 2)}
                    </div>
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))
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
