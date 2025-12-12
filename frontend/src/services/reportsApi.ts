import api from './api'

export interface Report {
  id: string
  cycle_id: string
  discovery_id: string
  summary: string
  tasks_completed: number
  findings_count: number
  hypotheses_count: number
  papers_count: number
  budget_used: number
  generation_cost: number
  created_at: string
  is_final_report?: boolean
}

interface CycleReportsResponse {
  cycle_reports: Report[]
  count: number
}

interface CycleReportContentResponse {
  full_content: string
}

export const reportsApi = {
  // Get all cycle reports for a discovery
  async getReports(discoveryId: string): Promise<Report[]> {
    const { data } = await api.get<CycleReportsResponse>(
      `/api/v1/reports/${discoveryId}/cycle-reports`
    )
    return data.cycle_reports || []
  },

  // Get cycle report content
  async getReportContent(discoveryId: string, cycleId: string): Promise<string> {
    const { data } = await api.get<CycleReportContentResponse>(
      `/api/v1/reports/${discoveryId}/cycle-reports/${cycleId}`
    )
    return data.full_content
  },

  // Generate a new report
  async generateReport(discoveryId: string, reportType: string): Promise<Report> {
    const { data } = await api.post<Report>(`/api/v1/reports/${discoveryId}/generate`, {
      report_type: reportType
    })
    return data
  }
}
