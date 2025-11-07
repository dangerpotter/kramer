import api from './api'
import type {
  DiscoveryConfig,
  DiscoveryResponse,
  DiscoveryDetail,
  CycleInfo,
  MetricsResponse,
} from '@/types/discovery'

export const discoveryApi = {
  // Create and start a new discovery
  async startDiscovery(config: DiscoveryConfig): Promise<DiscoveryResponse> {
    const { data } = await api.post<DiscoveryResponse>('/api/v1/discovery/start', config)
    return data
  },

  // Get discovery status
  async getStatus(discoveryId: string): Promise<any> {
    const { data } = await api.get(`/api/v1/discovery/${discoveryId}/status`)
    return data
  },

  // Stop a running discovery
  async stopDiscovery(discoveryId: string): Promise<void> {
    await api.post(`/api/v1/discovery/${discoveryId}/stop`)
  },

  // Get all cycles
  async getCycles(discoveryId: string): Promise<CycleInfo[]> {
    const { data } = await api.get<CycleInfo[]>(`/api/v1/discovery/${discoveryId}/cycles`)
    return data
  },

  // Get metrics
  async getMetrics(discoveryId: string): Promise<MetricsResponse> {
    const { data } = await api.get<MetricsResponse>(`/api/v1/discovery/${discoveryId}/metrics`)
    return data
  },

  // List all discoveries
  async listDiscoveries(): Promise<any[]> {
    const { data } = await api.get('/api/v1/discovery/')
    return data
  },
}
