import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useStartDiscovery } from '@/hooks/useDiscovery'
import Card from '@/components/common/Card'
import { Loader2, Play } from 'lucide-react'
import type { DiscoveryConfig } from '@/types/discovery'

export default function Configure() {
  const navigate = useNavigate()
  const startDiscovery = useStartDiscovery()

  const [config, setConfig] = useState<DiscoveryConfig>({
    objective: '',
    dataset_path: '',
    max_cycles: 20,
    max_total_budget: 100,
    max_parallel_tasks: 4,
    enable_checkpointing: true,
    checkpoint_interval: 5,
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    try {
      const response = await startDiscovery.mutateAsync(config)
      navigate(`/dashboard/${response.discovery_id}`)
    } catch (error) {
      console.error('Failed to start discovery:', error)
      alert('Failed to start discovery. Please try again.')
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">
        Configure New Discovery
      </h1>

      <form onSubmit={handleSubmit} className="space-y-6">
        <Card title="Research Objective" subtitle="Describe what you want to discover">
          <textarea
            value={config.objective}
            onChange={(e) => setConfig({ ...config, objective: e.target.value })}
            className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                     focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            rows={4}
            placeholder="Example: Investigate the relationship between protein folding and disease mechanisms in Alzheimer's..."
            required
          />
        </Card>

        <Card title="Dataset" subtitle="Optional: Upload or specify a dataset path">
          <input
            type="text"
            value={config.dataset_path}
            onChange={(e) => setConfig({ ...config, dataset_path: e.target.value })}
            className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                     focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="Path to dataset file or leave empty"
          />
        </Card>

        <Card title="Parameters" subtitle="Configure discovery constraints">
          <div className="grid grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Max Cycles
              </label>
              <input
                type="number"
                value={config.max_cycles}
                onChange={(e) => setConfig({ ...config, max_cycles: parseInt(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                         focus:ring-2 focus:ring-primary-500"
                min={1}
                max={100}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Budget (USD)
              </label>
              <input
                type="number"
                value={config.max_total_budget}
                onChange={(e) => setConfig({ ...config, max_total_budget: parseFloat(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                         focus:ring-2 focus:ring-primary-500"
                min={0}
                step={0.01}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Max Parallel Tasks
              </label>
              <input
                type="number"
                value={config.max_parallel_tasks}
                onChange={(e) => setConfig({ ...config, max_parallel_tasks: parseInt(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                         focus:ring-2 focus:ring-primary-500"
                min={1}
                max={10}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Checkpoint Interval
              </label>
              <input
                type="number"
                value={config.checkpoint_interval}
                onChange={(e) => setConfig({ ...config, checkpoint_interval: parseInt(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                         focus:ring-2 focus:ring-primary-500"
                min={1}
              />
            </div>
          </div>

          <div className="mt-4">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={config.enable_checkpointing}
                onChange={(e) => setConfig({ ...config, enable_checkpointing: e.target.checked })}
                className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Enable automatic checkpointing
              </span>
            </label>
          </div>
        </Card>

        <button
          type="submit"
          disabled={startDiscovery.isPending || !config.objective}
          className="w-full flex items-center justify-center space-x-2 px-6 py-3
                   bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg
                   disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {startDiscovery.isPending ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Starting Discovery...</span>
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              <span>Start Discovery</span>
            </>
          )}
        </button>
      </form>
    </div>
  )
}
