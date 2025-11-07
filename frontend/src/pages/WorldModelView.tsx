import { useParams } from 'react-router-dom'
import { useState } from 'react'
import GraphVisualization from '@/components/visualization/GraphVisualization'
import GraphControls from '@/components/visualization/GraphControls'
import LegendPanel from '@/components/visualization/LegendPanel'

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
        <div className="lg:col-span-3">
          <GraphVisualization
            discoveryId={discoveryId}
            nodeTypeFilter={nodeTypeFilter}
            edgeTypeFilter={edgeTypeFilter}
            layout={layout}
          />
        </div>

        <div className="lg:col-span-1">
          <LegendPanel />
        </div>
      </div>
    </div>
  )
}
