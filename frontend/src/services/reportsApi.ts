import api from './api'

export interface Report {
  id: string
  name: string
  discovery_id: string
  created_at: string
  file_path?: string
}

export interface ReportContent {
  content: string
  metadata?: Record<string, any>
}

interface ReportsResponse {
  reports: Report[]
  count: number
}

export const reportsApi = {
  // Get all reports for a discovery
  async getReports(discoveryId: string): Promise<Report[]> {
    const { data } = await api.get<ReportsResponse>(`/api/v1/reports/${discoveryId}`)
    return data.reports || []
  },

  // Get report content
  async getReportContent(discoveryId: string, reportId: string): Promise<string> {
    const { data } = await api.get<ReportContent>(
      `/api/v1/reports/${discoveryId}/${reportId}`
    )
    return data.content
  },

  // Generate a new report
  async generateReport(discoveryId: string, reportType: string): Promise<Report> {
    const { data } = await api.post<Report>(`/api/v1/reports/${discoveryId}/generate`, {
      report_type: reportType
    })
    return data
  }
}
