import React from 'react'

export const CardSkeleton = () => (
  <div className="animate-pulse p-4 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800">
    <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-3/4 mb-3"></div>
    <div className="h-8 bg-gray-300 dark:bg-gray-600 rounded mb-2"></div>
    <div className="h-8 bg-gray-300 dark:bg-gray-600 rounded"></div>
  </div>
)

export const TableSkeleton = () => (
  <div className="space-y-3 animate-pulse">
    {[...Array(5)].map((_, i) => (
      <div key={i} className="h-12 bg-gray-300 dark:bg-gray-600 rounded"></div>
    ))}
  </div>
)

export const ListSkeleton = () => (
  <div className="space-y-2 animate-pulse">
    {[...Array(3)].map((_, i) => (
      <div key={i} className="flex items-center gap-3 p-3 border border-gray-300 dark:border-gray-600 rounded-lg">
        <div className="w-10 h-10 bg-gray-300 dark:bg-gray-600 rounded-full"></div>
        <div className="flex-1 space-y-2">
          <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-3/4"></div>
          <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-1/2"></div>
        </div>
      </div>
    ))}
  </div>
)

export const GraphSkeleton = () => (
  <div className="animate-pulse">
    <div className="w-full h-[700px] border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
      <div className="text-gray-400 dark:text-gray-500">Loading graph...</div>
    </div>
  </div>
)

export const ChartSkeleton = () => (
  <div className="animate-pulse p-4 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800">
    <div className="h-6 bg-gray-300 dark:bg-gray-600 rounded w-1/3 mb-4"></div>
    <div className="h-[300px] bg-gray-200 dark:bg-gray-700 rounded"></div>
  </div>
)
