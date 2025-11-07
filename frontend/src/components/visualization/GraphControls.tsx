import React from 'react'
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'

interface GraphControlsProps {
  onLayoutChange: (layout: string) => void
  onZoomFit: () => void
  onResetZoom: () => void
  nodeTypeFilter: string[]
  onNodeTypeFilterChange: (filter: string[]) => void
  edgeTypeFilter: string[]
  onEdgeTypeFilterChange: (filter: string[]) => void
}

export default function GraphControls({
  onLayoutChange,
  onZoomFit,
  onResetZoom,
  nodeTypeFilter,
  onNodeTypeFilterChange,
  edgeTypeFilter,
  onEdgeTypeFilterChange
}: GraphControlsProps) {
  const layouts = ['cose', 'circle', 'grid', 'breadthfirst', 'concentric']
  const nodeTypes = ['HYPOTHESIS', 'FINDING', 'PAPER', 'DATASET', 'QUESTION']
  const edgeTypes = ['SUPPORTS', 'REFUTES', 'DERIVES_FROM', 'RELATES_TO']

  const toggleNodeType = (type: string) => {
    if (nodeTypeFilter.includes(type)) {
      onNodeTypeFilterChange(nodeTypeFilter.filter(t => t !== type))
    } else {
      onNodeTypeFilterChange([...nodeTypeFilter, type])
    }
  }

  const toggleEdgeType = (type: string) => {
    if (edgeTypeFilter.includes(type)) {
      onEdgeTypeFilterChange(edgeTypeFilter.filter(t => t !== type))
    } else {
      onEdgeTypeFilterChange([...edgeTypeFilter, type])
    }
  }

  return (
    <div className="flex flex-wrap gap-4 items-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      {/* Layout selector */}
      <div className="flex items-center gap-2">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Layout:
        </label>
        <select
          onChange={(e) => onLayoutChange(e.target.value)}
          className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
        >
          {layouts.map(layout => (
            <option key={layout} value={layout}>
              {layout.charAt(0).toUpperCase() + layout.slice(1)}
            </option>
          ))}
        </select>
      </div>

      {/* Zoom controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={onZoomFit}
          className="flex items-center gap-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm transition-colors"
          title="Fit to Screen"
        >
          <Maximize2 className="w-4 h-4" />
          <span>Fit</span>
        </button>
        <button
          onClick={onResetZoom}
          className="flex items-center gap-1 px-3 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-md text-sm transition-colors"
          title="Reset Zoom"
        >
          <ZoomOut className="w-4 h-4" />
          <span>Reset</span>
        </button>
      </div>

      {/* Node type filters */}
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Nodes:</span>
        <div className="flex flex-wrap gap-2">
          {nodeTypes.map(type => (
            <label
              key={type}
              className="flex items-center gap-1 px-2 py-1 rounded-md bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 text-xs cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-600"
            >
              <input
                type="checkbox"
                checked={nodeTypeFilter.length === 0 || nodeTypeFilter.includes(type)}
                onChange={() => toggleNodeType(type)}
                className="rounded border-gray-300 dark:border-gray-600"
              />
              <span className="text-gray-700 dark:text-gray-300">{type}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Edge type filters */}
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Edges:</span>
        <div className="flex flex-wrap gap-2">
          {edgeTypes.map(type => (
            <label
              key={type}
              className="flex items-center gap-1 px-2 py-1 rounded-md bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 text-xs cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-600"
            >
              <input
                type="checkbox"
                checked={edgeTypeFilter.length === 0 || edgeTypeFilter.includes(type)}
                onChange={() => toggleEdgeType(type)}
                className="rounded border-gray-300 dark:border-gray-600"
              />
              <span className="text-gray-700 dark:text-gray-300">{type}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  )
}
