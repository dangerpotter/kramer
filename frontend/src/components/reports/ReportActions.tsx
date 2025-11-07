import React, { useState } from 'react'
import { Download, Copy, CheckCircle } from 'lucide-react'
import toast from 'react-hot-toast'

interface ReportActionsProps {
  reportId: string | null
  content: string | null
}

export default function ReportActions({ reportId, content }: ReportActionsProps) {
  const [copied, setCopied] = useState(false)

  const handleDownload = () => {
    if (!content || !reportId) return

    const blob = new Blob([content], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `report-${reportId}.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    toast.success('Report downloaded successfully')
  }

  const handleCopy = async () => {
    if (!content) return

    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      toast.success('Report copied to clipboard')

      setTimeout(() => {
        setCopied(false)
      }, 2000)
    } catch (error) {
      toast.error('Failed to copy to clipboard')
    }
  }

  if (!reportId || !content) {
    return null
  }

  return (
    <div className="flex gap-2">
      <button
        onClick={handleDownload}
        className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
      >
        <Download className="w-4 h-4" />
        <span>Download Markdown</span>
      </button>
      <button
        onClick={handleCopy}
        className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${
          copied
            ? 'bg-green-600 text-white'
            : 'bg-gray-600 hover:bg-gray-700 text-white'
        }`}
      >
        {copied ? (
          <>
            <CheckCircle className="w-4 h-4" />
            <span>Copied!</span>
          </>
        ) : (
          <>
            <Copy className="w-4 h-4" />
            <span>Copy to Clipboard</span>
          </>
        )}
      </button>
    </div>
  )
}
