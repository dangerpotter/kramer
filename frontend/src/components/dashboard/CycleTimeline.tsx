import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface CycleData {
  cycle_number: number
  duration_seconds?: number
  tasks_completed?: number
  findings_added?: number
}

interface CycleTimelineProps {
  cycles: CycleData[]
}

export default function CycleTimeline({ cycles }: CycleTimelineProps) {
  if (!cycles || cycles.length === 0) {
    return (
      <div className="p-4 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800">
        <h3 className="font-bold text-lg mb-4 text-gray-900 dark:text-white">Cycle Performance</h3>
        <div className="flex items-center justify-center h-[300px] text-gray-500 dark:text-gray-400">
          No cycle data available
        </div>
      </div>
    )
  }

  const data = cycles.map(cycle => ({
    cycle: cycle.cycle_number,
    duration: cycle.duration_seconds ? cycle.duration_seconds / 60 : 0, // Convert to minutes
    tasks: cycle.tasks_completed || 0,
    findings: cycle.findings_added || 0
  }))

  return (
    <div className="p-4 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800">
      <h3 className="font-bold text-lg mb-4 text-gray-900 dark:text-white">Cycle Performance</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="cycle"
            label={{ value: 'Cycle', position: 'insideBottom', offset: -5 }}
            stroke="#9ca3af"
          />
          <YAxis
            label={{ value: 'Count / Minutes', angle: -90, position: 'insideLeft' }}
            stroke="#9ca3af"
          />
          <Tooltip
            formatter={(value: number, name: string) => {
              if (name === 'Duration (min)') return `${value.toFixed(1)} min`
              return value
            }}
            labelFormatter={(label) => `Cycle ${label}`}
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '0.5rem',
              color: '#f9fafb'
            }}
          />
          <Legend />
          <Bar dataKey="duration" fill="#3b82f6" name="Duration (min)" />
          <Bar dataKey="tasks" fill="#10b981" name="Tasks" />
          <Bar dataKey="findings" fill="#f59e0b" name="Findings" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
