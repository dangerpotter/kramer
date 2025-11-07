import { useParams } from 'react-router-dom'
import Card from '@/components/common/Card'

export default function Reports() {
  const { discoveryId } = useParams<{ discoveryId: string }>()

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
        Discovery Reports
      </h1>

      <Card>
        <div className="text-gray-500 dark:text-gray-400">
          Report viewer with markdown rendering will be implemented here.
          <br />
          Discovery ID: {discoveryId}
        </div>
      </Card>
    </div>
  )
}
