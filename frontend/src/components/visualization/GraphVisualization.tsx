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
            label: truncate(node.text, 50),
            type: node.node_type,
            confidence: node.confidence || 0,
            fullData: node
          }
        })),

        edges: filteredEdges.map(edge => ({
          data: {
            id: edge.edge_id,
            source: edge.source_id,
            target: edge.target_id,
            type: edge.edge_type,
            label: edge.edge_type
          }
        }))
      },

      style: [
        // Node styles by type
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'text-wrap': 'wrap',
            'text-max-width': '100px',
            'font-size': '10px',
            'width': 40,
            'height': 40,
            'text-valign': 'center',
            'text-halign': 'center',
          }
        },
        {
          selector: 'node[type="HYPOTHESIS"]',
          style: {
            'background-color': '#3b82f6',
            'shape': 'diamond',
            'width': 50,
            'height': 50,
          }
        },
        {
          selector: 'node[type="FINDING"]',
          style: {
            'background-color': '#10b981',
            'shape': 'ellipse',
          }
        },
        {
          selector: 'node[type="PAPER"]',
          style: {
            'background-color': '#f59e0b',
            'shape': 'rectangle',
          }
        },
        {
          selector: 'node[type="DATASET"]',
          style: {
            'background-color': '#8b5cf6',
            'shape': 'barrel',
          }
        },
        {
          selector: 'node[type="QUESTION"]',
          style: {
            'background-color': '#ec4899',
            'shape': 'round-octagon',
          }
        },

        // Edge styles by type
        {
          selector: 'edge',
          style: {
            'width': 2,
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'label': 'data(label)',
            'font-size': '8px',
            'text-rotation': 'autorotate',
          }
        },
        {
          selector: 'edge[type="SUPPORTS"]',
          style: {
            'line-color': '#10b981',
            'target-arrow-color': '#10b981',
          }
        },
        {
          selector: 'edge[type="REFUTES"]',
          style: {
            'line-color': '#ef4444',
            'target-arrow-color': '#ef4444',
            'line-style': 'dashed',
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
            'border-width': 3,
            'border-color': '#1e40af',
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

  return (
    <div className="relative">
      <div
        ref={containerRef}
        className="w-full h-[700px] border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
      />

      {selectedNode && (
        <NodeDetailPanel node={selectedNode} onClose={() => setSelectedNode(null)} />
      )}
    </div>
  )
}
