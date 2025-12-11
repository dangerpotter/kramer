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
        <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
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
              value: 'Cost (USD)',
              angle: -90,
              position: 'insideLeft',
              offset: 10,
              style: { textAnchor: 'middle', fill: '#9ca3af', fontSize: 12 }
            }}
            tick={{ fontSize: 11, fill: '#9ca3af' }}
            stroke="#9ca3af"
            width={60}
          />
          <Tooltip
            formatter={(value: number) => `$${value.toFixed(2)}`}
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
            cursor={{ stroke: '#6b7280', strokeWidth: 1 }}
          />
          <Legend
            verticalAlign="top"
            align="right"
            wrapperStyle={{
              paddingBottom: 20,
              fontSize: 12
            }}
          />
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
