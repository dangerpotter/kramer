import { useQuery } from '@tanstack/react-query'
import { worldModelApi } from '@/services/worldModelApi'

export function useWorldModelGraph(discoveryId: string, nodeType?: string) {
  return useQuery({
    queryKey: ['world-model', discoveryId, 'graph', nodeType],
    queryFn: () => worldModelApi.getGraph(discoveryId, nodeType),
    enabled: !!discoveryId,
  })
}

export function useNode(discoveryId: string, nodeId: string) {
  return useQuery({
    queryKey: ['world-model', discoveryId, 'node', nodeId],
    queryFn: () => worldModelApi.getNode(discoveryId, nodeId),
    enabled: !!discoveryId && !!nodeId,
  })
}

export function useFindings(discoveryId: string, minConfidence: number = 0.0) {
  return useQuery({
    queryKey: ['world-model', discoveryId, 'findings', minConfidence],
    queryFn: () => worldModelApi.getFindings(discoveryId, minConfidence),
    enabled: !!discoveryId,
  })
}

export function useHypotheses(discoveryId: string, testedOnly: boolean = false) {
  return useQuery({
    queryKey: ['world-model', discoveryId, 'hypotheses', testedOnly],
    queryFn: () => worldModelApi.getHypotheses(discoveryId, testedOnly),
    enabled: !!discoveryId,
  })
}

export function usePapers(discoveryId: string) {
  return useQuery({
    queryKey: ['world-model', discoveryId, 'papers'],
    queryFn: () => worldModelApi.getPapers(discoveryId),
    enabled: !!discoveryId,
  })
}
