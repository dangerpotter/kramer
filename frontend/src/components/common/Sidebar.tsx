import { Link, useLocation } from 'react-router-dom'
import { Home, Settings, FileSearch, Network, FileText, Plus } from 'lucide-react'
import { cn } from '@/utils/cn'

const navItems = [
  { name: 'New Discovery', path: '/configure', icon: Plus },
  { name: 'Dashboard', path: '/dashboard', icon: Home, requiresId: true },
  { name: 'Explorer', path: '/explorer', icon: FileSearch, requiresId: true },
  { name: 'World Model', path: '/world-model', icon: Network, requiresId: true },
  { name: 'Reports', path: '/reports', icon: FileText, requiresId: true },
]

export default function Sidebar() {
  const location = useLocation()

  return (
    <aside className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700">
      <div className="flex flex-col h-full">
        <div className="p-6">
          <div className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-primary-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">K</span>
            </div>
            <span className="text-xl font-bold text-gray-900 dark:text-white">Kramer</span>
          </div>
        </div>

        <nav className="flex-1 px-4 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname.startsWith(item.path)

            return (
              <Link
                key={item.path}
                to={item.path}
                className={cn(
                  'flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors',
                  isActive
                    ? 'bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                )}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{item.name}</span>
              </Link>
            )
          })}
        </nav>

        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          <div className="text-sm text-gray-500 dark:text-gray-400">
            Version 1.0.0
          </div>
        </div>
      </div>
    </aside>
  )
}
