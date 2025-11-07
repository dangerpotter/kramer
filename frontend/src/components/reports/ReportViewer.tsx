import React from 'react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useQuery } from '@tanstack/react-query'
import { reportsApi } from '@/services/reportsApi'
import Loading from '@/components/common/Loading'
import { FileQuestion } from 'lucide-react'

interface ReportViewerProps {
  discoveryId: string
  reportId: string | null
  onContentLoaded?: (content: string) => void
}

export default function ReportViewer({
  discoveryId,
  reportId,
  onContentLoaded
}: ReportViewerProps) {
  const { data: content, isLoading, error } = useQuery({
    queryKey: ['report-content', discoveryId, reportId],
    queryFn: () => {
      if (!reportId) return null
      return reportsApi.getReportContent(discoveryId, reportId)
    },
    enabled: !!reportId && !!discoveryId
  })

  // Call the callback when content is loaded
  React.useEffect(() => {
    if (content && onContentLoaded) {
      onContentLoaded(content)
    }
  }, [content, onContentLoaded])

  if (!reportId) {
    return (
      <div className="text-center text-gray-500 dark:text-gray-400 py-12">
        <FileQuestion className="w-16 h-16 mx-auto mb-4 opacity-50" />
        <p>Select a report to view</p>
      </div>
    )
  }

  if (isLoading) return <Loading message="Loading report..." />

  if (error) {
    return (
      <div className="p-4 text-red-600 dark:text-red-400">
        Error loading report: {(error as Error).message}
      </div>
    )
  }

  if (!content) {
    return (
      <div className="text-center text-gray-500 dark:text-gray-400 py-12">
        <p>No content available</p>
      </div>
    )
  }

  return (
    <div className="prose prose-slate dark:prose-invert max-w-none">
      <ReactMarkdown
        components={{
          code({ node, inline, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '')
            return !inline && match ? (
              <SyntaxHighlighter
                style={vscDarkPlus}
                language={match[1]}
                PreTag="div"
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            )
          },
          // Custom heading styling
          h1: ({ node, ...props }) => (
            <h1 className="text-3xl font-bold mb-4 text-gray-900 dark:text-white" {...props} />
          ),
          h2: ({ node, ...props }) => (
            <h2 className="text-2xl font-bold mt-8 mb-3 text-gray-900 dark:text-white" {...props} />
          ),
          h3: ({ node, ...props }) => (
            <h3 className="text-xl font-bold mt-6 mb-2 text-gray-900 dark:text-white" {...props} />
          ),
          // Custom paragraph styling
          p: ({ node, ...props }) => (
            <p className="mb-4 text-gray-700 dark:text-gray-300" {...props} />
          ),
          // Custom list styling
          ul: ({ node, ...props }) => (
            <ul className="list-disc list-inside mb-4 space-y-1 text-gray-700 dark:text-gray-300" {...props} />
          ),
          ol: ({ node, ...props }) => (
            <ol className="list-decimal list-inside mb-4 space-y-1 text-gray-700 dark:text-gray-300" {...props} />
          ),
          // Custom link styling
          a: ({ node, ...props }) => (
            <a className="text-blue-600 dark:text-blue-400 hover:underline" {...props} />
          ),
          // Custom blockquote styling
          blockquote: ({ node, ...props }) => (
            <blockquote className="border-l-4 border-gray-300 dark:border-gray-600 pl-4 italic text-gray-600 dark:text-gray-400" {...props} />
          ),
          // Custom table styling
          table: ({ node, ...props }) => (
            <div className="overflow-x-auto mb-4">
              <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-600" {...props} />
            </div>
          ),
          th: ({ node, ...props }) => (
            <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 bg-gray-100 dark:bg-gray-800" {...props} />
          ),
          td: ({ node, ...props }) => (
            <td className="border border-gray-300 dark:border-gray-600 px-4 py-2" {...props} />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
