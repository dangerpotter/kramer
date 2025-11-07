import { Search, Bell, User } from 'lucide-react'

export default function Header() {
  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center flex-1">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Kramer Discovery Platform
          </h1>
        </div>

        <div className="flex items-center space-x-4">
          <button className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
            <Bell className="w-5 h-5" />
          </button>
          <button className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
            <User className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  )
}
