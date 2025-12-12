import { useState } from 'react'
import { History, Search, Filter } from 'lucide-react'
import { useDiscoveries } from '@/hooks/useDiscovery'
import DiscoveryCard from '@/components/discovery/DiscoveryCard'
import { DiscoveryStatus } from '@/types/discovery'

type StatusFilter = 'all' | DiscoveryStatus

export default function DiscoveryHistory() {
  const { data: discoveries, isLoading, error } = useDiscoveries()
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [searchQuery, setSearchQuery] = useState('')

  // Filter discoveries by status and search query
  const filteredDiscoveries = discoveries?.filter((discovery) => {
    const matchesStatus = statusFilter === 'all' || discovery.status === statusFilter
    const matchesSearch =
      searchQuery === '' ||
      discovery.objective.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesStatus && matchesSearch
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-800 dark:text-red-200">
          Failed to load discoveries. Please try again later.
        </p>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <History className="w-8 h-8 text-primary-600" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Discovery History
          </h1>
        </div>
        <p className="text-gray-600 dark:text-gray-400">
          Browse and review your past discovery sessions
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search by objective..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Status Filter */}
        <div className="relative">
          <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
            className="pl-10 pr-8 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent appearance-none cursor-pointer"
          >
            <option value="all">All Status</option>
            <option value={DiscoveryStatus.RUNNING}>Running</option>
            <option value={DiscoveryStatus.COMPLETED}>Completed</option>
            <option value={DiscoveryStatus.FAILED}>Failed</option>
            <option value={DiscoveryStatus.STOPPED}>Stopped</option>
            <option value={DiscoveryStatus.PENDING}>Pending</option>
          </select>
        </div>
      </div>

      {/* Discovery List */}
      {filteredDiscoveries && filteredDiscoveries.length > 0 ? (
        <div className="grid gap-4">
          {filteredDiscoveries.map((discovery) => (
            <DiscoveryCard key={discovery.discovery_id} discovery={discovery} />
          ))}
        </div>
      ) : (
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8 text-center">
          <History className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            {discoveries?.length === 0 ? 'No discoveries yet' : 'No matching discoveries'}
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            {discoveries?.length === 0
              ? 'Start a new discovery to see it appear here.'
              : 'Try adjusting your search or filter criteria.'}
          </p>
        </div>
      )}

      {/* Stats footer */}
      {discoveries && discoveries.length > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex flex-wrap gap-4 text-sm text-gray-500 dark:text-gray-400">
            <span>Total: {discoveries.length} discoveries</span>
            <span>
              Completed: {discoveries.filter((d) => d.status === DiscoveryStatus.COMPLETED).length}
            </span>
            <span>
              Running: {discoveries.filter((d) => d.status === DiscoveryStatus.RUNNING).length}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
