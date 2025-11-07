import { useParams } from 'react-router-dom'
import Card from '@/components/common/Card'

export default function WorldModelView() {
  const { discoveryId } = useParams<{ discoveryId: string }>()

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
        World Model Visualization
      </h1>

      <Card>
        <div className="flex items-center justify-center h-96 text-gray-500 dark:text-gray-400">
          Graph visualization with Cytoscape.js will be implemented here.
          <br />
          Discovery ID: {discoveryId}
        </div>
      </Card>
    </div>
  )
}
