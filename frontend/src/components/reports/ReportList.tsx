import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { reportsApi } from '@/services/reportsApi'
import Loading from '@/components/common/Loading'
import { FileText, Clock, Plus, Loader2, Lightbulb, BookOpen, Award } from 'lucide-react'

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
  const queryClient = useQueryClient()
  const [showOptions, setShowOptions] = useState(false)

  const { data: reports, isLoading, error } = useQuery({
    queryKey: ['reports', discoveryId],
    queryFn: () => reportsApi.getReports(discoveryId),
    enabled: !!discoveryId
  })

  const generateMutation = useMutation({
    mutationFn: (reportType: string) => reportsApi.generateReport(discoveryId, reportType),
    onSuccess: (newReport) => {
      queryClient.invalidateQueries({ queryKey: ['reports', discoveryId] })
      onSelectReport(newReport.report_id || newReport.id)
      setShowOptions(false)
    }
  })

  const handleGenerate = (reportType: string) => {
    generateMutation.mutate(reportType)
  }

  if (isLoading) return <Loading message="Loading reports..." />

  if (error) {
    return (
      <div className="p-4 text-red-600 dark:text-red-400">
        Error loading reports: {(error as Error).message}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Generate Report Button */}
      <div className="relative">
        <button
          onClick={() => setShowOptions(!showOptions)}
          disabled={generateMutation.isPending}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white rounded-lg transition-colors"
        >
          {generateMutation.isPending ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Plus className="w-5 h-5" />
              Generate Report
            </>
          )}
        </button>

        {/* Dropdown options */}
        {showOptions && !generateMutation.isPending && (
          <div className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-10">
            <button
              onClick={() => handleGenerate('summary')}
              className="w-full text-left px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 border-b border-gray-200 dark:border-gray-700"
            >
              <div className="font-medium text-gray-900 dark:text-white">Summary Report</div>
              <div className="text-sm text-gray-500 dark:text-gray-400">Key findings and hypotheses</div>
            </button>
            <button
              onClick={() => handleGenerate('detailed')}
              className="w-full text-left px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 border-b border-gray-200 dark:border-gray-700"
            >
              <div className="font-medium text-gray-900 dark:text-white">Detailed Report</div>
              <div className="text-sm text-gray-500 dark:text-gray-400">Full analysis with all evidence</div>
            </button>
            <button
              onClick={() => handleGenerate('executive')}
              className="w-full text-left px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-b-lg"
            >
              <div className="font-medium text-gray-900 dark:text-white">Executive Summary</div>
              <div className="text-sm text-gray-500 dark:text-gray-400">High-level overview</div>
            </button>
          </div>
        )}
      </div>

      {generateMutation.isError && (
        <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-600 dark:text-red-400">
          Failed to generate report: {(generateMutation.error as Error).message}
        </div>
      )}

      {/* Report List */}
      <div className="space-y-2">
        <h3 className="font-bold text-lg text-gray-900 dark:text-white px-2">
          Generated Reports
        </h3>

        <div className="max-h-[60vh] overflow-y-auto space-y-2 pr-1">
        {(!reports || reports.length === 0) ? (
          <div className="p-4 text-gray-500 dark:text-gray-400 text-center">
            <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No reports generated yet</p>
            <p className="text-xs mt-1">Reports are generated after each discovery cycle</p>
          </div>
        ) : (
          reports.map(report => (
            <button
              key={report.id}
              onClick={() => onSelectReport(report.cycle_id)}
              className={`w-full text-left p-3 border rounded-lg transition-colors ${
                selectedReportId === report.cycle_id
                  ? report.is_final_report
                    ? 'bg-amber-50 dark:bg-amber-900/30 border-amber-500 dark:border-amber-400'
                    : 'bg-blue-50 dark:bg-blue-900 border-blue-500 dark:border-blue-400'
                  : report.is_final_report
                    ? 'bg-amber-50/50 dark:bg-amber-900/10 border-amber-300 dark:border-amber-600 hover:bg-amber-50 dark:hover:bg-amber-900/20'
                    : 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              <div className="flex items-start gap-2">
                {report.is_final_report ? (
                  <Award className="w-5 h-5 text-amber-500 dark:text-amber-400 mt-0.5 flex-shrink-0" />
                ) : (
                  <FileText className="w-5 h-5 text-gray-500 dark:text-gray-400 mt-0.5 flex-shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className={`font-medium ${report.is_final_report ? 'text-amber-700 dark:text-amber-300' : 'text-gray-900 dark:text-white'}`}>
                    {report.is_final_report ? 'Final Report' : 'Cycle Report'}
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 line-clamp-2">
                    {report.summary}
                  </p>
                  {!report.is_final_report && (
                    <div className="flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400 mt-2">
                      <span className="flex items-center gap-1">
                        <Lightbulb className="w-3 h-3" />
                        {report.findings_count} findings
                      </span>
                      <span className="flex items-center gap-1">
                        <BookOpen className="w-3 h-3" />
                        {report.hypotheses_count} hypotheses
                      </span>
                    </div>
                  )}
                  <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400 mt-1">
                    <Clock className="w-3 h-3" />
                    <span>{new Date(report.created_at).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </button>
          ))
        )}
        </div>
      </div>
    </div>
  )
}
