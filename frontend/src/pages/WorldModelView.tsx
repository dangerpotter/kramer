import { useParams } from 'react-router-dom'
import { useState } from 'react'
import GraphVisualization from '@/components/visualization/GraphVisualization'
import GraphControls from '@/components/visualization/GraphControls'
import LegendPanel from '@/components/visualization/LegendPanel'
import { useFindings, useHypotheses, usePapers, useGraph } from '@/hooks/useWorldModel'
import { useDiscoveryMetrics } from '@/hooks/useDiscovery'

// Insights Panel component
function InsightsPanel({ discoveryId }: { discoveryId: string }) {
  const { data: findings } = useFindings(discoveryId)
  const { data: hypotheses } = useHypotheses(discoveryId)
  const { data: papers } = usePapers(discoveryId)
  const { data: graphData } = useGraph(discoveryId)
  const { data: metrics } = useDiscoveryMetrics(discoveryId)

  const supportedHypotheses = hypotheses?.filter(h => h.status === 'supported') || []
  const refutedHypotheses = hypotheses?.filter(h => h.status === 'refuted') || []
  const untestedHypotheses = hypotheses?.filter(h => h.status === 'untested' || !h.status) || []

  const highConfidenceFindings = findings?.filter(f => f.confidence >= 0.7) || []
  const mediumConfidenceFindings = findings?.filter(f => f.confidence >= 0.4 && f.confidence < 0.7) || []
  const lowConfidenceFindings = findings?.filter(f => f.confidence < 0.4) || []

  // Get cycle count from metrics API (actual cycles run, not unique cycles in data)
  const cycleCount = metrics?.current_cycle || 0

  // Count relationships
  const supportsCount = graphData?.edges?.filter(e => e.edge_type === 'supports').length || 0
  const refutesCount = graphData?.edges?.filter(e => e.edge_type === 'refutes').length || 0
  const derivesCount = graphData?.edges?.filter(e => e.edge_type === 'derives_from').length || 0

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 space-y-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
        Research Insights
      </h3>

      {/* Progress Overview */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Progress Overview</h4>
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-gray-50 dark:bg-gray-700 rounded p-3 text-center">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">{cycleCount}</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Cycles Run</div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded p-3 text-center">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">{papers?.length || 0}</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Papers Found</div>
          </div>
        </div>
      </div>

      {/* Hypothesis Status */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Hypothesis Status</h4>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="text-sm text-gray-600 dark:text-gray-400">Supported</span>
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">{supportedHypotheses.length}</span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span className="text-sm text-gray-600 dark:text-gray-400">Refuted</span>
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">{refutedHypotheses.length}</span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gray-400"></div>
              <span className="text-sm text-gray-600 dark:text-gray-400">Untested</span>
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">{untestedHypotheses.length}</span>
          </div>
        </div>
        {/* Progress bar */}
        {hypotheses && hypotheses.length > 0 && (
          <div className="w-full h-2 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden flex">
            <div
              className="bg-green-500 h-full"
              style={{ width: `${(supportedHypotheses.length / hypotheses.length) * 100}%` }}
            />
            <div
              className="bg-red-500 h-full"
              style={{ width: `${(refutedHypotheses.length / hypotheses.length) * 100}%` }}
            />
          </div>
        )}
      </div>

      {/* Finding Confidence */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Finding Confidence</h4>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400">High (≥70%)</span>
            <span className="text-sm font-medium text-green-600">{highConfidenceFindings.length}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400">Medium (40-70%)</span>
            <span className="text-sm font-medium text-yellow-600">{mediumConfidenceFindings.length}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400">Low (&lt;40%)</span>
            <span className="text-sm font-medium text-red-600">{lowConfidenceFindings.length}</span>
          </div>
        </div>
      </div>

      {/* Relationships */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Relationships</h4>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400">Supports</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">{supportsCount}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400">Refutes</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">{refutesCount}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400">Derives From</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">{derivesCount}</span>
          </div>
        </div>
      </div>

      {/* Top Hypotheses */}
      {supportedHypotheses.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-green-700 dark:text-green-400">Top Supported Hypotheses</h4>
          <div className="space-y-2">
            {supportedHypotheses.slice(0, 3).map((h, i) => (
              <div key={h.hypothesis_id} className="p-2 bg-green-50 dark:bg-green-900/20 rounded text-sm">
                <p className="text-gray-800 dark:text-gray-200 line-clamp-2">{h.text}</p>
                {h.confidence !== undefined && h.confidence > 0 && (
                  <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                    Confidence: {(h.confidence * 100).toFixed(0)}%
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Refuted Hypotheses */}
      {refutedHypotheses.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-red-700 dark:text-red-400">Refuted Hypotheses</h4>
          <div className="space-y-2">
            {refutedHypotheses.slice(0, 2).map((h, i) => (
              <div key={h.hypothesis_id} className="p-2 bg-red-50 dark:bg-red-900/20 rounded text-sm">
                <p className="text-gray-800 dark:text-gray-200 line-clamp-2">{h.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* High Confidence Findings */}
      {highConfidenceFindings.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">High Confidence Findings</h4>
          <div className="space-y-2">
            {highConfidenceFindings.slice(0, 3).map((f, i) => (
              <div key={f.finding_id} className="p-2 bg-gray-50 dark:bg-gray-700 rounded text-sm">
                <p className="text-gray-800 dark:text-gray-200 line-clamp-2">{f.text}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Confidence: {(f.confidence * 100).toFixed(0)}%
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default function WorldModelView() {
  const { discoveryId } = useParams<{ discoveryId: string }>()
  const [layout, setLayout] = useState('cose')
  const [nodeTypeFilter, setNodeTypeFilter] = useState<string[]>([])
  const [edgeTypeFilter, setEdgeTypeFilter] = useState<string[]>([])

  const handleLayoutChange = (newLayout: string) => {
    setLayout(newLayout)
    // Trigger layout change via global method
    if ((window as any).graphControls) {
      (window as any).graphControls.changeLayout(newLayout)
    }
  }

  const handleZoomFit = () => {
    if ((window as any).graphControls) {
      (window as any).graphControls.zoomToFit()
    }
  }

  const handleResetZoom = () => {
    if ((window as any).graphControls) {
      (window as any).graphControls.resetZoom()
    }
  }

  if (!discoveryId) {
    return (
      <div className="p-6">
        <div className="text-red-600 dark:text-red-400">
          Error: Discovery ID is required
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
        Knowledge Graph Visualization
      </h1>

      <GraphControls
        onLayoutChange={handleLayoutChange}
        onZoomFit={handleZoomFit}
        onResetZoom={handleResetZoom}
        nodeTypeFilter={nodeTypeFilter}
        onNodeTypeFilterChange={setNodeTypeFilter}
        edgeTypeFilter={edgeTypeFilter}
        onEdgeTypeFilterChange={setEdgeTypeFilter}
      />

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left side - Legend */}
        <div className="lg:col-span-1 order-2 lg:order-1">
          <LegendPanel />
        </div>

        {/* Center - Graph */}
        <div className="lg:col-span-2 order-1 lg:order-2">
          <GraphVisualization
            discoveryId={discoveryId}
            nodeTypeFilter={nodeTypeFilter}
            edgeTypeFilter={edgeTypeFilter}
            layout={layout}
          />
        </div>

        {/* Right side - Insights */}
        <div className="lg:col-span-1 order-3">
          <InsightsPanel discoveryId={discoveryId} />
        </div>
      </div>
    </div>
  )
}
