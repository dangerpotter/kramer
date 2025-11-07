import { useParams } from 'react-router-dom'
import { useState } from 'react'
import ReportList from '@/components/reports/ReportList'
import ReportViewer from '@/components/reports/ReportViewer'
import ReportActions from '@/components/reports/ReportActions'

export default function Reports() {
  const { discoveryId } = useParams<{ discoveryId: string }>()
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null)
  const [reportContent, setReportContent] = useState<string | null>(null)

  if (!discoveryId) {
    return (
      <div className="p-6">
        <div className="text-red-600 dark:text-red-400">
          Error: Discovery ID is required
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
        Discovery Reports
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar with report list */}
        <div className="lg:col-span-1">
          <div className="sticky top-6">
            <ReportList
              discoveryId={discoveryId}
              onSelectReport={setSelectedReportId}
              selectedReportId={selectedReportId}
            />
          </div>
        </div>

        {/* Main content area */}
        <div className="lg:col-span-3 space-y-4">
          {selectedReportId && (
            <ReportActions
              reportId={selectedReportId}
              content={reportContent}
            />
          )}

          <div className="border border-gray-300 dark:border-gray-600 rounded-lg p-6 bg-white dark:bg-gray-800">
            <ReportViewer
              discoveryId={discoveryId}
              reportId={selectedReportId}
              onContentLoaded={setReportContent}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
