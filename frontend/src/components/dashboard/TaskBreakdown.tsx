import React from 'react'
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface Task {
  task_type: string
  [key: string]: any
}

interface TaskBreakdownProps {
  tasks: Task[]
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4']

// Format task type: "search_literature" → "Search Literature"
const formatTaskType = (type: string): string => {
  return type
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ')
}

export default function TaskBreakdown({ tasks }: TaskBreakdownProps) {
  if (!tasks || tasks.length === 0) {
    return (
      <div className="p-4 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800">
        <h3 className="font-bold text-lg mb-4 text-gray-900 dark:text-white">Task Distribution</h3>
        <div className="flex items-center justify-center h-[300px] text-gray-500 dark:text-gray-400">
          No task data available
        </div>
      </div>
    )
  }

  const taskCounts = tasks.reduce((acc: Record<string, number>, task) => {
    const taskType = task.task_type || 'UNKNOWN'
    acc[taskType] = (acc[taskType] || 0) + 1
    return acc
  }, {})

  const data = Object.entries(taskCounts).map(([type, count]) => ({
    name: formatTaskType(type),
    value: count as number
  }))

  const renderCustomizedLabel = ({
    cx,
    cy,
    midAngle,
    innerRadius,
    outerRadius,
    percent
  }: any) => {
    // Only show label for slices >= 5% to prevent bleeding
    if (percent < 0.05) return null

    const RADIAN = Math.PI / 180
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5
    const x = cx + radius * Math.cos(-midAngle * RADIAN)
    const y = cy + radius * Math.sin(-midAngle * RADIAN)

    return (
      <text
        x={x}
        y={y}
        fill="white"
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="11"
        fontWeight="600"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    )
  }

  return (
    <div className="p-4 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800">
      <h3 className="font-bold text-lg mb-4 text-gray-900 dark:text-white">Task Distribution</h3>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={renderCustomizedLabel}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              border: '1px solid #e5e7eb',
              borderRadius: '0.5rem',
              color: '#1f2937',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
            }}
            itemStyle={{ color: '#374151' }}
          />
          <Legend
            verticalAlign="bottom"
            height={36}
            wrapperStyle={{ color: '#9ca3af' }}
          />
        </PieChart>
      </ResponsiveContainer>

      {/* Summary */}
      <div className="mt-4 pt-4 border-t border-gray-300 dark:border-gray-600">
        <div className="grid grid-cols-2 gap-2 text-sm">
          {data.map((item, index) => (
            <div key={item.name} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded"
                style={{ backgroundColor: COLORS[index % COLORS.length] }}
              />
              <span className="text-gray-700 dark:text-gray-300">
                {item.name}: <span className="font-semibold">{item.value}</span>
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
