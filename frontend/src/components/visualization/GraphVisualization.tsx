import React, { useEffect, useRef, useState } from 'react'
import cytoscape from 'cytoscape'
import { useWorldModelGraph } from '@/hooks/useWorldModel'
import type { GraphNode } from '@/types/worldModel'
import Loading from '@/components/common/Loading'
import NodeDetailPanel from './NodeDetailPanel'

interface GraphVisualizationProps {
  discoveryId: string
  nodeTypeFilter?: string[]
  edgeTypeFilter?: string[]
  layout?: string
  onZoomFit?: () => void
  onResetZoom?: () => void
}

const truncate = (text: string, maxLength: number) => {
  if (text.length <= maxLength) return text
  return text.substring(0, maxLength) + '...'
}

export default function GraphVisualization({
  discoveryId,
  nodeTypeFilter = [],
  edgeTypeFilter = [],
  layout = 'cose'
}: GraphVisualizationProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<cytoscape.Core | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)

  const { data: graphData, isLoading } = useWorldModelGraph(discoveryId)

  useEffect(() => {
    if (!graphData || !containerRef.current) return

    // Filter nodes and edges
    const filteredNodes = graphData.nodes.filter(
      n => nodeTypeFilter.length === 0 || nodeTypeFilter.includes(n.node_type)
    )
    const filteredEdges = graphData.edges.filter(
      e => edgeTypeFilter.length === 0 || edgeTypeFilter.includes(e.edge_type)
    )

    // Initialize Cytoscape
    const cy = cytoscape({
      container: containerRef.current,

      elements: {
        nodes: filteredNodes.map(node => ({
          data: {
            id: node.node_id,
            label: truncate(node.text, 40),
            type: node.node_type.toUpperCase(), // Normalize to uppercase for selectors
            confidence: node.confidence || 0,
            fullData: node
          }
        })),

        edges: filteredEdges.map(edge => ({
          data: {
            id: edge.edge_id,
            source: edge.source_id,
            target: edge.target_id,
            type: edge.edge_type.toUpperCase(), // Normalize to uppercase for selectors
            label: edge.edge_type.toLowerCase().replace('_', ' ')
          }
        }))
      },

      style: [
        // Base node styles
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'text-wrap': 'wrap',
            'text-max-width': '120px',
            'font-size': '11px',
            'font-weight': 'bold',
            'color': '#ffffff',
            'text-outline-color': '#000000',
            'text-outline-width': 1,
            'width': 45,
            'height': 45,
            'text-valign': 'bottom',
            'text-halign': 'center',
            'text-margin-y': 8,
            'border-width': 2,
            'border-color': '#ffffff',
            'border-opacity': 0.8,
            'background-opacity': 0.9,
            'transition-property': 'background-color, border-color, width, height',
            'transition-duration': '0.2s',
          }
        },
        {
          selector: 'node[type="HYPOTHESIS"]',
          style: {
            'background-color': '#3b82f6',
            'shape': 'diamond',
            'width': 55,
            'height': 55,
            'border-color': '#1d4ed8',
          }
        },
        {
          selector: 'node[type="FINDING"]',
          style: {
            'background-color': '#10b981',
            'shape': 'ellipse',
            'border-color': '#059669',
          }
        },
        {
          selector: 'node[type="PAPER"]',
          style: {
            'background-color': '#f59e0b',
            'shape': 'rectangle',
            'border-color': '#d97706',
          }
        },
        {
          selector: 'node[type="DATASET"]',
          style: {
            'background-color': '#8b5cf6',
            'shape': 'barrel',
            'border-color': '#7c3aed',
          }
        },
        {
          selector: 'node[type="QUESTION"]',
          style: {
            'background-color': '#ec4899',
            'shape': 'round-octagon',
            'border-color': '#db2777',
          }
        },

        // Hover effect
        {
          selector: 'node:active',
          style: {
            'overlay-opacity': 0.2,
            'overlay-color': '#ffffff',
          }
        },

        // Edge styles
        {
          selector: 'edge',
          style: {
            'width': 2,
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 1.2,
            'label': 'data(label)',
            'font-size': '9px',
            'text-rotation': 'autorotate',
            'text-background-color': '#1f2937',
            'text-background-opacity': 0.8,
            'text-background-padding': '3px',
            'color': '#e5e7eb',
            'opacity': 0.8,
            'transition-property': 'width, opacity',
            'transition-duration': '0.2s',
          }
        },
        {
          selector: 'edge[type="SUPPORTS"]',
          style: {
            'line-color': '#10b981',
            'target-arrow-color': '#10b981',
            'width': 3,
          }
        },
        {
          selector: 'edge[type="REFUTES"]',
          style: {
            'line-color': '#ef4444',
            'target-arrow-color': '#ef4444',
            'line-style': 'dashed',
            'width': 3,
          }
        },
        {
          selector: 'edge[type="DERIVES_FROM"]',
          style: {
            'line-color': '#6b7280',
            'target-arrow-color': '#6b7280',
          }
        },
        {
          selector: 'edge[type="RELATES_TO"]',
          style: {
            'line-color': '#94a3b8',
            'target-arrow-color': '#94a3b8',
            'line-style': 'dotted',
          }
        },

        // Selected node
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': '#fbbf24',
            'background-opacity': 1,
          }
        },

        // Connected edges glow on node select
        {
          selector: 'edge:selected',
          style: {
            'width': 4,
            'opacity': 1,
          }
        },
      ],

      layout: {
        name: layout,
        idealEdgeLength: 100,
        nodeOverlap: 20,
        refresh: 20,
        fit: true,
        padding: 30,
        randomize: false,
        componentSpacing: 100,
        nodeRepulsion: 400000,
        edgeElasticity: 100,
        nestingFactor: 5,
        gravity: 80,
        numIter: 1000,
        initialTemp: 200,
        coolingFactor: 0.95,
        minTemp: 1.0
      },

      minZoom: 0.3,
      maxZoom: 3,
    })

    // Event handlers
    cy.on('tap', 'node', (event) => {
      const node = event.target
      setSelectedNode(node.data('fullData'))
    })

    cy.on('tap', (event) => {
      if (event.target === cy) {
        setSelectedNode(null)
      }
    })

    cyRef.current = cy

    return () => {
      cy.destroy()
    }
  }, [graphData, nodeTypeFilter, edgeTypeFilter, layout, discoveryId])

  // Expose methods for external controls
  useEffect(() => {
    if (!cyRef.current) return

    // Store methods on window for external access
    (window as any).graphControls = {
      zoomToFit: () => {
        cyRef.current?.fit(undefined, 50)
      },
      resetZoom: () => {
        cyRef.current?.zoom(1)
        cyRef.current?.center()
      },
      changeLayout: (layoutName: string) => {
        cyRef.current?.layout({ name: layoutName } as any).run()
      }
    }
  }, [])

  if (isLoading) return <Loading message="Loading graph..." />

  // Legend items for the graph
  const legendItems = [
    { type: 'HYPOTHESIS', color: '#3b82f6', shape: 'diamond', label: 'Hypothesis' },
    { type: 'FINDING', color: '#10b981', shape: 'circle', label: 'Finding' },
    { type: 'PAPER', color: '#f59e0b', shape: 'square', label: 'Paper' },
    { type: 'DATASET', color: '#8b5cf6', shape: 'pill', label: 'Dataset' },
    { type: 'QUESTION', color: '#ec4899', shape: 'octagon', label: 'Question' },
  ]

  const edgeLegend = [
    { type: 'SUPPORTS', color: '#10b981', style: 'solid', label: 'Supports' },
    { type: 'REFUTES', color: '#ef4444', style: 'dashed', label: 'Refutes' },
    { type: 'DERIVES_FROM', color: '#6b7280', style: 'solid', label: 'Derives From' },
    { type: 'RELATES_TO', color: '#94a3b8', style: 'dotted', label: 'Relates To' },
  ]

  return (
    <div className="relative">
      <div
        ref={containerRef}
        className="w-full h-[700px] border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-900"
      />

      {/* Legend */}
      <div className="absolute top-4 left-4 bg-gray-800/90 backdrop-blur-sm rounded-lg p-3 border border-gray-600 shadow-lg">
        <div className="text-xs font-semibold text-gray-300 mb-2">Node Types</div>
        <div className="space-y-1.5">
          {legendItems.map(item => (
            <div key={item.type} className="flex items-center gap-2">
              <div
                className={`w-4 h-4 border-2 ${
                  item.shape === 'diamond' ? 'rotate-45' : ''
                } ${item.shape === 'circle' ? 'rounded-full' : ''} ${
                  item.shape === 'square' ? 'rounded-sm' : ''
                } ${item.shape === 'pill' ? 'rounded-full w-5' : ''} ${
                  item.shape === 'octagon' ? 'rounded' : ''
                }`}
                style={{ backgroundColor: item.color, borderColor: item.color }}
              />
              <span className="text-xs text-gray-300">{item.label}</span>
            </div>
          ))}
        </div>
        <div className="border-t border-gray-600 mt-2 pt-2">
          <div className="text-xs font-semibold text-gray-300 mb-2">Relationships</div>
          <div className="space-y-1.5">
            {edgeLegend.map(item => (
              <div key={item.type} className="flex items-center gap-2">
                <div className="w-5 h-0.5 relative">
                  <div
                    className={`absolute inset-0 ${
                      item.style === 'dashed' ? 'border-t-2 border-dashed' :
                      item.style === 'dotted' ? 'border-t-2 border-dotted' : ''
                    }`}
                    style={{
                      backgroundColor: item.style === 'solid' ? item.color : 'transparent',
                      borderColor: item.color
                    }}
                  />
                </div>
                <span className="text-xs text-gray-300">{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {selectedNode && (
        <NodeDetailPanel node={selectedNode} onClose={() => setSelectedNode(null)} />
      )}
    </div>
  )
}
