import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import ErrorBoundary from '@/components/common/ErrorBoundary'
import Header from '@/components/common/Header'
import Sidebar from '@/components/common/Sidebar'
import Dashboard from '@/pages/Dashboard'
import Configure from '@/pages/Configure'
import Explorer from '@/pages/Explorer'
import WorldModelView from '@/pages/WorldModelView'
import Reports from '@/pages/Reports'

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
          <Sidebar />
          <div className="flex flex-col flex-1 overflow-hidden">
            <Header />
            <main className="flex-1 overflow-y-auto p-6">
              <Routes>
                <Route path="/" element={<Navigate to="/configure" replace />} />
                <Route path="/configure" element={<Configure />} />
                <Route path="/dashboard/:discoveryId" element={<Dashboard />} />
                <Route path="/explorer/:discoveryId" element={<Explorer />} />
                <Route path="/world-model/:discoveryId" element={<WorldModelView />} />
                <Route path="/reports/:discoveryId" element={<Reports />} />
              </Routes>
            </main>
          </div>
        </div>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

export default App
