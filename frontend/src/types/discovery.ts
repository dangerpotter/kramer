export interface DiscoveryConfig {
  objective: string
  dataset_path?: string
  max_cycles: number
  max_total_budget: number
  max_parallel_tasks: number
  enable_checkpointing: boolean
  checkpoint_interval: number
}

export enum DiscoveryStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  STOPPED = 'stopped',
}

export interface DiscoveryResponse {
  discovery_id: string
  status: DiscoveryStatus
  message?: string
}

export interface TaskStatusInfo {
  task_id: string
  task_type: string
  status: string
  objective: string
  created_at: string
  started_at?: string
  completed_at?: string
}

export interface CycleInfo {
  cycle_id: string
  cycle_number: number
  status: string
  tasks: TaskStatusInfo[]
  budget_used: number
  findings_generated: number
  hypotheses_generated: number
  created_at: string
  started_at?: string
  completed_at?: string
}

export interface DiscoveryDetail {
  discovery_id: string
  objective: string
  status: DiscoveryStatus
  config: DiscoveryConfig
  current_cycle: number
  total_cycles: number
  total_cost: number
  findings_count: number
  hypotheses_count: number
  papers_count: number
  created_at: string
  started_at?: string
  completed_at?: string
}

export interface MetricsResponse {
  discovery_id: string
  current_cycle: number
  total_cost: number
  cost_per_cycle: number[]
  findings_per_cycle: number[]
  hypotheses_per_cycle: number[]
  tasks_completed: number
  tasks_pending: number
  tasks_running: number
  avg_task_duration: number
  estimated_time_remaining?: number
}
