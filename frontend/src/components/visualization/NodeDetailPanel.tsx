import React from 'react'
import { X } from 'lucide-react'
import type { GraphNode } from '@/types/worldModel'

interface NodeDetailPanelProps {
  node: GraphNode
  onClose: () => void
}

export default function NodeDetailPanel({ node, onClose }: NodeDetailPanelProps) {
  return (
    <div className="absolute top-4 right-4 w-80 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 shadow-lg rounded-lg p-4">
      <div className="flex justify-between items-start mb-3">
        <h3 className="font-bold text-lg text-gray-900 dark:text-white">{node.node_type}</h3>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          aria-label="Close"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="space-y-3 text-sm">
        <div>
          <span className="font-medium text-gray-700 dark:text-gray-300">ID:</span>
          <p className="text-gray-600 dark:text-gray-400 font-mono text-xs mt-1 break-all">
            {node.node_id}
          </p>
        </div>

        <div>
          <span className="font-medium text-gray-700 dark:text-gray-300">Text:</span>
          <p className="mt-1 text-gray-700 dark:text-gray-300">{node.text}</p>
        </div>

        {node.confidence !== undefined && (
          <div>
            <span className="font-medium text-gray-700 dark:text-gray-300">Confidence:</span>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full"
                  style={{ width: `${node.confidence * 100}%` }}
                />
              </div>
              <span className="text-gray-600 dark:text-gray-400">
                {(node.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        )}

        {node.metadata && Object.keys(node.metadata).length > 0 && (
          <div>
            <span className="font-medium text-gray-700 dark:text-gray-300">Metadata:</span>
            <div className="mt-1 space-y-1">
              {Object.entries(node.metadata).map(([key, value]) => (
                <div key={key} className="text-xs">
                  <span className="text-gray-600 dark:text-gray-400">{key}:</span>{' '}
                  <span className="text-gray-700 dark:text-gray-300">
                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div>
          <span className="font-medium text-gray-700 dark:text-gray-300">Created:</span>
          <p className="text-gray-600 dark:text-gray-400 text-xs mt-1">
            {new Date(node.created_at).toLocaleString()}
          </p>
        </div>
      </div>
    </div>
  )
}
