export interface GraphNode {
  node_id: string
  node_type: string
  text: string
  metadata: Record<string, any>
  confidence?: number
  created_at: string
}

export interface GraphEdge {
  edge_id: string
  source_id: string
  target_id: string
  edge_type: string
  metadata: Record<string, any>
  created_at: string
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
  node_count: number
  edge_count: number
}

export interface Finding {
  finding_id: string
  text: string
  confidence: number
  supporting_evidence: string[]
  source?: string
  cycle_discovered: number
  metadata: Record<string, any>
  created_at: string
}

export interface Hypothesis {
  hypothesis_id: string
  text: string
  confidence?: number
  status: string
  supporting_findings: string[]
  refuting_findings: string[]
  test_results?: Record<string, any>
  cycle_generated: number
  metadata: Record<string, any>
  created_at: string
}

export interface Paper {
  paper_id: string
  title: string
  authors: string[]
  abstract?: string
  url?: string
  year?: number
  relevance_score?: number
  key_findings: string[]
  metadata: Record<string, any>
  created_at: string
}
