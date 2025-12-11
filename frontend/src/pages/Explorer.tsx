import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { useFindings, useHypotheses, usePapers } from '@/hooks/useWorldModel'
import Card from '@/components/common/Card'
import Loading from '@/components/common/Loading'
import { Finding, Hypothesis, Paper } from '@/types/worldModel'

// Confidence bar component
function ConfidenceBar({ confidence }: { confidence: number }) {
  const percentage = Math.round(confidence * 100)
  const getColor = () => {
    if (percentage >= 70) return 'bg-green-500'
    if (percentage >= 40) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${getColor()} transition-all`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-sm text-gray-600 dark:text-gray-400">{percentage}%</span>
    </div>
  )
}

// Badge component for sources and types
function Badge({ children, variant = 'default' }: { children: React.ReactNode, variant?: 'default' | 'success' | 'warning' | 'error' | 'info' }) {
  const variants = {
    default: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
    success: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
    warning: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300',
    error: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300',
    info: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
  }

  return (
    <span className={`px-2 py-1 rounded text-xs font-medium ${variants[variant]}`}>
      {children}
    </span>
  )
}

// Enhanced Finding Card
function FindingCard({ finding }: { finding: Finding }) {
  const [expanded, setExpanded] = useState(false)

  const getSourceBadge = () => {
    const source = finding.metadata?.evidence_type || finding.source || 'unknown'
    if (source === 'literature_review') return <Badge variant="info">Literature</Badge>
    if (source === 'statistical_analysis') return <Badge variant="success">Data Analysis</Badge>
    if (source === 'hypothesis_test') return <Badge variant="warning">Hypothesis Test</Badge>
    return <Badge>{source}</Badge>
  }

  return (
    <Card>
      <div className="space-y-3">
        <div className="flex items-start justify-between gap-4">
          <p className="text-gray-900 dark:text-white flex-1">{finding.text}</p>
          {getSourceBadge()}
        </div>

        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <span className="text-gray-500 dark:text-gray-400">Confidence:</span>
            <ConfidenceBar confidence={finding.confidence} />
          </div>
          <span className="text-gray-500 dark:text-gray-400">
            Cycle {finding.cycle_discovered}
          </span>
        </div>

        {/* Metadata section */}
        {finding.metadata && Object.keys(finding.metadata).length > 0 && (
          <div>
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-sm text-primary-500 hover:text-primary-600 flex items-center gap-1"
            >
              {expanded ? '▼' : '▶'} Details
            </button>
            {expanded && (
              <div className="mt-2 p-3 bg-gray-50 dark:bg-gray-800 rounded text-sm">
                {finding.metadata.from_hypothesis_test && (
                  <p className="text-gray-600 dark:text-gray-400">
                    <span className="font-medium">From hypothesis:</span> {finding.metadata.from_hypothesis_test.slice(0, 8)}...
                  </p>
                )}
                {finding.metadata.paper_title && (
                  <p className="text-gray-600 dark:text-gray-400">
                    <span className="font-medium">Paper:</span> {finding.metadata.paper_title}
                  </p>
                )}
                {finding.metadata.verdict && (
                  <p className="text-gray-600 dark:text-gray-400">
                    <span className="font-medium">Verdict:</span> {finding.metadata.verdict}
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </Card>
  )
}

// Enhanced Hypothesis Card
function HypothesisCard({ hypothesis }: { hypothesis: Hypothesis }) {
  const [expanded, setExpanded] = useState(false)

  const getStatusBadge = () => {
    switch (hypothesis.status) {
      case 'supported':
        return <Badge variant="success">Supported</Badge>
      case 'refuted':
        return <Badge variant="error">Refuted</Badge>
      case 'testing':
        return <Badge variant="warning">Testing</Badge>
      default:
        return <Badge>Untested</Badge>
    }
  }

  const getTestTypeBadge = () => {
    const testType = hypothesis.metadata?.test_type || hypothesis.test_results?.test_type
    if (!testType) return null

    const labels: Record<string, string> = {
      'comprehensive': 'Comprehensive Test',
      'literature_based': 'Literature Review',
      'data_driven': 'Data Analysis',
    }
    return <Badge variant="info">{labels[testType] || testType}</Badge>
  }

  return (
    <Card>
      <div className="space-y-3">
        <div className="flex items-start justify-between gap-4">
          <p className="text-gray-900 dark:text-white flex-1 font-medium">{hypothesis.text}</p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {getStatusBadge()}
          {getTestTypeBadge()}
          {hypothesis.confidence !== undefined && hypothesis.confidence > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-500 dark:text-gray-400">Confidence:</span>
              <ConfidenceBar confidence={hypothesis.confidence} />
            </div>
          )}
          <span className="text-sm text-gray-500 dark:text-gray-400">
            Cycle {hypothesis.cycle_generated}
          </span>
        </div>

        {/* Test results and evidence */}
        {(hypothesis.test_results || hypothesis.supporting_findings?.length > 0 || hypothesis.refuting_findings?.length > 0) && (
          <div>
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-sm text-primary-500 hover:text-primary-600 flex items-center gap-1"
            >
              {expanded ? '▼' : '▶'} Test Results & Evidence
            </button>
            {expanded && (
              <div className="mt-2 p-3 bg-gray-50 dark:bg-gray-800 rounded space-y-3">
                {/* Reasoning */}
                {hypothesis.test_results?.reasoning && (
                  <div>
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Reasoning:</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {hypothesis.test_results.reasoning}
                    </p>
                  </div>
                )}

                {/* Supporting findings */}
                {hypothesis.supporting_findings?.length > 0 && (
                  <div>
                    <p className="text-sm font-medium text-green-700 dark:text-green-400">
                      Supporting Evidence ({hypothesis.supporting_findings.length}):
                    </p>
                    <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {hypothesis.supporting_findings.slice(0, 3).map((id, i) => (
                        <li key={i}>Finding {id.slice(0, 8)}...</li>
                      ))}
                      {hypothesis.supporting_findings.length > 3 && (
                        <li>...and {hypothesis.supporting_findings.length - 3} more</li>
                      )}
                    </ul>
                  </div>
                )}

                {/* Refuting findings */}
                {hypothesis.refuting_findings?.length > 0 && (
                  <div>
                    <p className="text-sm font-medium text-red-700 dark:text-red-400">
                      Refuting Evidence ({hypothesis.refuting_findings.length}):
                    </p>
                    <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {hypothesis.refuting_findings.slice(0, 3).map((id, i) => (
                        <li key={i}>Finding {id.slice(0, 8)}...</li>
                      ))}
                      {hypothesis.refuting_findings.length > 3 && (
                        <li>...and {hypothesis.refuting_findings.length - 3} more</li>
                      )}
                    </ul>
                  </div>
                )}

                {/* Metadata */}
                {hypothesis.metadata?.rationale && (
                  <div>
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Rationale:</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {hypothesis.metadata.rationale}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </Card>
  )
}

// Enhanced Paper Card
function PaperCard({ paper }: { paper: Paper }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <Card>
      <div className="space-y-3">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <h3 className="font-semibold text-gray-900 dark:text-white">
              {paper.url ? (
                <a
                  href={paper.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-primary-500 transition-colors"
                >
                  {paper.title}
                </a>
              ) : paper.title}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {paper.authors.join(', ')} {paper.year && `(${paper.year})`}
            </p>
          </div>
          {paper.relevance_score !== undefined && paper.relevance_score > 0 && (
            <div className="text-right">
              <span className="text-xs text-gray-500 dark:text-gray-400">Relevance</span>
              <div className="font-medium text-primary-600 dark:text-primary-400">
                {(paper.relevance_score * 100).toFixed(0)}%
              </div>
            </div>
          )}
        </div>

        {paper.abstract && (
          <div>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              {expanded ? paper.abstract : `${paper.abstract.substring(0, 200)}...`}
            </p>
            {paper.abstract.length > 200 && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="text-sm text-primary-500 hover:text-primary-600 mt-1"
              >
                {expanded ? 'Show less' : 'Read more'}
              </button>
            )}
          </div>
        )}

        {/* Key findings from paper */}
        {paper.key_findings?.length > 0 && (
          <div className="border-t border-gray-200 dark:border-gray-700 pt-2">
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Key Findings:</p>
            <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-400 mt-1">
              {paper.key_findings.map((finding, i) => (
                <li key={i}>{finding}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Source badge */}
        {paper.metadata?.source && (
          <div className="flex items-center gap-2">
            <Badge variant="info">{paper.metadata.source}</Badge>
            {paper.url && (
              <a
                href={paper.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-primary-500 hover:text-primary-600"
              >
                View Paper →
              </a>
            )}
          </div>
        )}
      </div>
    </Card>
  )
}

// Summary statistics component
function SummaryStats({
  findings,
  hypotheses,
  papers
}: {
  findings?: Finding[]
  hypotheses?: Hypothesis[]
  papers?: Paper[]
}) {
  const supportedCount = hypotheses?.filter(h => h.status === 'supported').length || 0
  const refutedCount = hypotheses?.filter(h => h.status === 'refuted').length || 0
  const untestedCount = hypotheses?.filter(h => h.status === 'untested' || !h.status).length || 0
  const avgConfidence = findings?.length
    ? (findings.reduce((acc, f) => acc + f.confidence, 0) / findings.length)
    : 0

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <p className="text-2xl font-bold text-gray-900 dark:text-white">{findings?.length || 0}</p>
        <p className="text-sm text-gray-500 dark:text-gray-400">Findings</p>
        <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
          Avg. confidence: {(avgConfidence * 100).toFixed(0)}%
        </p>
      </div>
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <p className="text-2xl font-bold text-green-600">{supportedCount}</p>
        <p className="text-sm text-gray-500 dark:text-gray-400">Supported</p>
      </div>
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <p className="text-2xl font-bold text-red-600">{refutedCount}</p>
        <p className="text-sm text-gray-500 dark:text-gray-400">Refuted</p>
      </div>
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <p className="text-2xl font-bold text-gray-600">{untestedCount}</p>
        <p className="text-sm text-gray-500 dark:text-gray-400">Untested</p>
      </div>
    </div>
  )
}

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

  const isLoading = loadingFindings || loadingHypotheses || loadingPapers

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
        Discovery Explorer
      </h1>

      {/* Summary statistics */}
      {!isLoading && (
        <SummaryStats findings={findings} hypotheses={hypotheses} papers={papers} />
      )}

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
              {findings?.length === 0 && (
                <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                  No findings yet. Run a discovery to generate findings.
                </p>
              )}
              {findings?.map((finding) => (
                <FindingCard key={finding.finding_id} finding={finding} />
              ))}
            </div>
          )
        )}

        {activeTab === 'hypotheses' && (
          loadingHypotheses ? <Loading /> : (
            <div className="space-y-4">
              {hypotheses?.length === 0 && (
                <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                  No hypotheses yet. Run a discovery to generate hypotheses.
                </p>
              )}
              {hypotheses?.map((hypothesis) => (
                <HypothesisCard key={hypothesis.hypothesis_id} hypothesis={hypothesis} />
              ))}
            </div>
          )
        )}

        {activeTab === 'papers' && (
          loadingPapers ? <Loading /> : (
            <div className="space-y-4">
              {papers?.length === 0 && (
                <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                  No papers found yet. Literature search will populate this list.
                </p>
              )}
              {papers?.map((paper) => (
                <PaperCard key={paper.paper_id} paper={paper} />
              ))}
            </div>
          )
        )}
      </div>
    </div>
  )
}
