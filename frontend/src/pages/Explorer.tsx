import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { useFindings, useHypotheses, usePapers } from '@/hooks/useWorldModel'
import Card from '@/components/common/Card'
import Loading from '@/components/common/Loading'

export default function Explorer() {
  const { discoveryId } = useParams<{ discoveryId: string }>()
  const [activeTab, setActiveTab] = useState<'findings' | 'hypotheses' | 'papers'>('findings')

  const { data: findings, isLoading: loadingFindings } = useFindings(discoveryId!)
  const { data: hypotheses, isLoading: loadingHypotheses } = useHypotheses(discoveryId!)
  const { data: papers, isLoading: loadingPapers } = usePapers(discoveryId!)

  const tabs = [
    { id: 'findings' as const, label: 'Findings', count: findings?.length || 0 },
    { id: 'hypotheses' as const, label: 'Hypotheses', count: hypotheses?.length || 0 },
    { id: 'papers' as const, label: 'Papers', count: papers?.length || 0 },
  ]

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
        Discovery Explorer
      </h1>

      <div className="border-b border-gray-200 dark:border-gray-700">
        <div className="flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400'
              }`}
            >
              {tab.label} ({tab.count})
            </button>
          ))}
        </div>
      </div>

      <div>
        {activeTab === 'findings' && (
          loadingFindings ? <Loading /> : (
            <div className="space-y-4">
              {findings?.map((finding) => (
                <Card key={finding.finding_id}>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="text-gray-900 dark:text-white">{finding.text}</p>
                      <div className="mt-2 flex items-center space-x-4 text-sm text-gray-500">
                        <span>Confidence: {(finding.confidence * 100).toFixed(0)}%</span>
                        <span>Cycle: {finding.cycle_discovered}</span>
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )
        )}

        {activeTab === 'hypotheses' && (
          loadingHypotheses ? <Loading /> : (
            <div className="space-y-4">
              {hypotheses?.map((hypothesis) => (
                <Card key={hypothesis.hypothesis_id}>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="text-gray-900 dark:text-white">{hypothesis.text}</p>
                      <div className="mt-2 flex items-center space-x-4 text-sm">
                        <span className={`px-2 py-1 rounded ${
                          hypothesis.status === 'supported' ? 'bg-green-100 text-green-800' :
                          hypothesis.status === 'refuted' ? 'bg-red-100 text-red-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {hypothesis.status}
                        </span>
                        <span className="text-gray-500">Cycle: {hypothesis.cycle_generated}</span>
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )
        )}

        {activeTab === 'papers' && (
          loadingPapers ? <Loading /> : (
            <div className="space-y-4">
              {papers?.map((paper) => (
                <Card key={paper.paper_id}>
                  <h3 className="font-semibold text-gray-900 dark:text-white">{paper.title}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {paper.authors.join(', ')} {paper.year && `(${paper.year})`}
                  </p>
                  {paper.abstract && (
                    <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                      {paper.abstract.substring(0, 200)}...
                    </p>
                  )}
                </Card>
              ))}
            </div>
          )
        )}
      </div>
    </div>
  )
}
