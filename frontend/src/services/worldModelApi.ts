import api from './api'
import type { GraphData, Finding, Hypothesis, Paper } from '@/types/worldModel'

export const worldModelApi = {
  // Get graph data
  async getGraph(discoveryId: string, nodeType?: string): Promise<GraphData> {
    const params = nodeType ? { node_type: nodeType } : {}
    const { data } = await api.get<GraphData>(
      `/api/v1/world-model/${discoveryId}/graph`,
      { params }
    )
    return data
  },

  // Get node details
  async getNode(discoveryId: string, nodeId: string): Promise<any> {
    const { data } = await api.get(`/api/v1/world-model/${discoveryId}/nodes/${nodeId}`)
    return data
  },

  // Get findings
  async getFindings(discoveryId: string, minConfidence: number = 0.0): Promise<Finding[]> {
    const { data } = await api.get<Finding[]>(
      `/api/v1/world-model/${discoveryId}/findings`,
      { params: { min_confidence: minConfidence } }
    )
    return data
  },

  // Get hypotheses
  async getHypotheses(discoveryId: string, testedOnly: boolean = false): Promise<Hypothesis[]> {
    const { data } = await api.get<Hypothesis[]>(
      `/api/v1/world-model/${discoveryId}/hypotheses`,
      { params: { tested_only: testedOnly } }
    )
    return data
  },

  // Get papers
  async getPapers(discoveryId: string): Promise<Paper[]> {
    const { data } = await api.get<Paper[]>(`/api/v1/world-model/${discoveryId}/papers`)
    return data
  },
}
