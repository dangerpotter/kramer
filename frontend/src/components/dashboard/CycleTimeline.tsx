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
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="cycle"
            label={{
              value: 'Cycle',
              position: 'insideBottom',
              offset: -10,
              style: { fill: '#9ca3af', fontSize: 12 }
            }}
            tick={{ fontSize: 11, fill: '#9ca3af' }}
            stroke="#9ca3af"
          />
          <YAxis
            label={{
              value: 'Count / Minutes',
              angle: -90,
              position: 'insideLeft',
              offset: 10,
              style: { textAnchor: 'middle', fill: '#9ca3af', fontSize: 12 }
            }}
            tick={{ fontSize: 11, fill: '#9ca3af' }}
            stroke="#9ca3af"
            width={70}
          />
          <Tooltip
            formatter={(value: number, name: string) => {
              if (name === 'Duration (min)') return [`${value.toFixed(1)} min`, name]
              return [value, name]
            }}
            labelFormatter={(label) => `Cycle ${label}`}
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #4b5563',
              borderRadius: '8px',
              padding: '10px 14px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)'
            }}
            labelStyle={{
              color: '#f9fafb',
              fontWeight: 'bold',
              marginBottom: 4
            }}
            itemStyle={{
              color: '#d1d5db',
              padding: '2px 0'
            }}
            cursor={{ fill: 'rgba(107, 114, 128, 0.2)' }}
          />
          <Legend
            verticalAlign="top"
            align="right"
            wrapperStyle={{
              paddingBottom: 20,
              fontSize: 12
            }}
          />
          <Bar dataKey="duration" fill="#3b82f6" name="Duration (min)" radius={[4, 4, 0, 0]} />
          <Bar dataKey="tasks" fill="#10b981" name="Tasks" radius={[4, 4, 0, 0]} />
          <Bar dataKey="findings" fill="#f59e0b" name="Findings" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
