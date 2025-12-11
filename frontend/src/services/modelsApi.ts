import { api } from './api'

export interface ModelInfo {
  id: string
  display_name: string
  created_at?: string
}

export interface ModelsResponse {
  models: ModelInfo[]
}

export const modelsApi = {
  async getModels(): Promise<ModelInfo[]> {
    const { data } = await api.get<ModelsResponse>('/api/v1/models')
    return data.models
  },
}
