import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { reportsApi } from '@/services/reportsApi'
import Loading from '@/components/common/Loading'
import { FileText, Clock } from 'lucide-react'

interface ReportListProps {
  discoveryId: string
  onSelectReport: (reportId: string) => void
  selectedReportId: string | null
}

export default function ReportList({
  discoveryId,
  onSelectReport,
  selectedReportId
}: ReportListProps) {
  const { data: reports, isLoading, error } = useQuery({
    queryKey: ['reports', discoveryId],
    queryFn: () => reportsApi.getReports(discoveryId),
    enabled: !!discoveryId
  })

  if (isLoading) return <Loading message="Loading reports..." />

  if (error) {
    return (
      <div className="p-4 text-red-600 dark:text-red-400">
        Error loading reports: {(error as Error).message}
      </div>
    )
  }

  if (!reports || reports.length === 0) {
    return (
      <div className="p-4 text-gray-500 dark:text-gray-400 text-center">
        <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
        <p className="text-sm">No reports available</p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <h3 className="font-bold text-lg mb-3 text-gray-900 dark:text-white px-2">
        Generated Reports
      </h3>
      {reports.map(report => (
        <button
          key={report.id}
          onClick={() => onSelectReport(report.id)}
          className={`w-full text-left p-3 border rounded-lg transition-colors ${
            selectedReportId === report.id
              ? 'bg-blue-50 dark:bg-blue-900 border-blue-500 dark:border-blue-400'
              : 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
          }`}
        >
          <div className="flex items-start gap-2">
            <FileText className="w-5 h-5 text-gray-500 dark:text-gray-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="font-medium text-gray-900 dark:text-white truncate">
                {report.name}
              </div>
              <div className="flex items-center gap-1 text-sm text-gray-500 dark:text-gray-400 mt-1">
                <Clock className="w-3 h-3" />
                <span>{new Date(report.created_at).toLocaleString()}</span>
              </div>
            </div>
          </div>
        </button>
      ))}
    </div>
  )
}
