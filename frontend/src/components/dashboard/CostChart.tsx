import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface CostDataPoint {
  cycle: number
  cost: number
  cumulative: number
}

interface CostChartProps {
  data: CostDataPoint[]
}

export default function CostChart({ data }: CostChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="p-4 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800">
        <h3 className="font-bold text-lg mb-4 text-gray-900 dark:text-white">Cost Over Time</h3>
        <div className="flex items-center justify-center h-[300px] text-gray-500 dark:text-gray-400">
          No cost data available
        </div>
      </div>
    )
  }

  return (
    <div className="p-4 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800">
      <h3 className="font-bold text-lg mb-4 text-gray-900 dark:text-white">Cost Over Time</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="cycle"
            label={{ value: 'Cycle', position: 'insideBottom', offset: -5 }}
            stroke="#9ca3af"
          />
          <YAxis
            label={{ value: 'Cost (USD)', angle: -90, position: 'insideLeft' }}
            stroke="#9ca3af"
          />
          <Tooltip
            formatter={(value: number) => `$${value.toFixed(2)}`}
            labelFormatter={(label) => `Cycle ${label}`}
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '0.5rem',
              color: '#f9fafb'
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="cost"
            stroke="#3b82f6"
            name="Per Cycle"
            strokeWidth={2}
            dot={{ fill: '#3b82f6', r: 4 }}
            activeDot={{ r: 6 }}
          />
          <Line
            type="monotone"
            dataKey="cumulative"
            stroke="#10b981"
            name="Cumulative"
            strokeWidth={2}
            dot={{ fill: '#10b981', r: 4 }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
