import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { discoveryApi } from '@/services/discoveryApi'
import type { DiscoveryConfig } from '@/types/discovery'

export function useDiscovery(discoveryId: string) {
  return useQuery({
    queryKey: ['discovery', discoveryId, 'status'],
    queryFn: () => discoveryApi.getStatus(discoveryId),
    refetchInterval: 5000, // Poll every 5 seconds
    enabled: !!discoveryId,
  })
}

export function useDiscoveryCycles(discoveryId: string) {
  return useQuery({
    queryKey: ['discovery', discoveryId, 'cycles'],
    queryFn: () => discoveryApi.getCycles(discoveryId),
    refetchInterval: 5000,
    enabled: !!discoveryId,
  })
}

export function useDiscoveryMetrics(discoveryId: string) {
  return useQuery({
    queryKey: ['discovery', discoveryId, 'metrics'],
    queryFn: () => discoveryApi.getMetrics(discoveryId),
    refetchInterval: 5000,
    enabled: !!discoveryId,
  })
}

export function useStartDiscovery() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (config: DiscoveryConfig) => discoveryApi.startDiscovery(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discoveries'] })
    },
  })
}

export function useStopDiscovery() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (discoveryId: string) => discoveryApi.stopDiscovery(discoveryId),
    onSuccess: (_, discoveryId) => {
      queryClient.invalidateQueries({ queryKey: ['discovery', discoveryId] })
    },
  })
}

export function useDiscoveries() {
  return useQuery({
    queryKey: ['discoveries'],
    queryFn: () => discoveryApi.listDiscoveries(),
    refetchInterval: 10000,
  })
}
