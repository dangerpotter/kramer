import React from 'react'

export default function LegendPanel() {
  const nodeTypes = [
    { type: 'HYPOTHESIS', color: '#3b82f6', shape: '◆', description: 'Research hypothesis' },
    { type: 'FINDING', color: '#10b981', shape: '●', description: 'Data-driven finding' },
    { type: 'PAPER', color: '#f59e0b', shape: '■', description: 'Research paper' },
    { type: 'DATASET', color: '#8b5cf6', shape: '▭', description: 'Data source' },
    { type: 'QUESTION', color: '#ec4899', shape: '⬢', description: 'Research question' },
  ]

  const edgeTypes = [
    { type: 'SUPPORTS', color: '#10b981', style: 'solid', description: 'Provides evidence for' },
    { type: 'REFUTES', color: '#ef4444', style: 'dashed', description: 'Contradicts' },
    { type: 'DERIVES_FROM', color: '#6b7280', style: 'solid', description: 'Based on' },
    { type: 'RELATES_TO', color: '#94a3b8', style: 'dotted', description: 'Connected to' },
  ]

  return (
    <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <h4 className="font-bold text-gray-900 dark:text-white mb-3">Legend</h4>

      {/* Node Types */}
      <div className="mb-4">
        <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Node Types:</p>
        <div className="space-y-2">
          {nodeTypes.map(({ type, color, shape, description }) => (
            <div key={type} className="flex items-start gap-2 text-sm">
              <span
                className="text-xl leading-none flex-shrink-0"
                style={{ color }}
              >
                {shape}
              </span>
              <div className="flex-1">
                <div className="font-medium text-gray-900 dark:text-white">{type}</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">{description}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Edge Types */}
      <div>
        <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Edge Types:</p>
        <div className="space-y-2">
          {edgeTypes.map(({ type, color, style, description }) => (
            <div key={type} className="flex items-start gap-2 text-sm">
              <div className="flex-shrink-0 mt-2">
                <svg width="20" height="2">
                  <line
                    x1="0"
                    y1="1"
                    x2="20"
                    y2="1"
                    stroke={color}
                    strokeWidth="2"
                    strokeDasharray={style === 'dashed' ? '3,3' : style === 'dotted' ? '1,2' : '0'}
                  />
                </svg>
              </div>
              <div className="flex-1">
                <div className="font-medium text-gray-900 dark:text-white">{type}</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">{description}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <p className="text-xs text-gray-600 dark:text-gray-400">
          Click nodes to view details. Use controls above to adjust layout and filters.
        </p>
      </div>
    </div>
  )
}
