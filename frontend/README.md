# Kramer Web UI - Frontend

Modern React + TypeScript frontend for the Kramer autonomous discovery system.

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **React Query** - Data fetching
- **React Router** - Routing
- **Cytoscape.js** - Graph visualization
- **Recharts** - Data charts
- **React Markdown** - Markdown rendering
- **React Hot Toast** - Notifications

## Project Structure

```
frontend/
├── src/
│   ├── components/           # React components
│   │   ├── common/          # Shared components
│   │   │   ├── Card.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Loading.tsx
│   │   │   ├── ErrorBoundary.tsx
│   │   │   └── Skeleton.tsx
│   │   ├── dashboard/       # Dashboard charts
│   │   │   ├── CostChart.tsx
│   │   │   ├── CycleTimeline.tsx
│   │   │   └── TaskBreakdown.tsx
│   │   ├── reports/         # Report viewer
│   │   │   ├── ReportList.tsx
│   │   │   ├── ReportViewer.tsx
│   │   │   └── ReportActions.tsx
│   │   └── visualization/   # Graph visualization
│   │       ├── GraphVisualization.tsx
│   │       ├── GraphControls.tsx
│   │       ├── NodeDetailPanel.tsx
│   │       └── LegendPanel.tsx
│   ├── pages/               # Page components
│   │   ├── Configure.tsx    # Discovery creation
│   │   ├── Dashboard.tsx    # Real-time monitoring
│   │   ├── Explorer.tsx     # Results browser
│   │   ├── WorldModelView.tsx  # Graph visualization
│   │   └── Reports.tsx      # Report viewer
│   ├── hooks/               # Custom React hooks
│   │   ├── useDiscovery.ts  # Discovery API
│   │   ├── useWorldModel.ts # World model API
│   │   └── useWebSocket.ts  # WebSocket connection
│   ├── services/            # API clients
│   │   ├── api.ts          # Axios instance
│   │   ├── discoveryApi.ts
│   │   ├── worldModelApi.ts
│   │   └── reportsApi.ts
│   ├── types/               # TypeScript types
│   │   ├── discovery.ts
│   │   └── worldModel.ts
│   ├── App.tsx             # Main app component
│   ├── main.tsx            # Entry point
│   └── index.css           # Global styles
├── public/                  # Static assets
├── .env.example            # Environment template
├── Dockerfile              # Production container
├── nginx.conf              # Nginx config
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

## Development Setup

### Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Copy environment template
cp .env.example .env

# Edit .env if needed (defaults should work)
# VITE_API_URL=http://localhost:8000
# VITE_WS_URL=ws://localhost:8000
```

### Available Scripts

```bash
# Start development server (http://localhost:3000)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run TypeScript type checking
npm run type-check

# Run linter
npm run lint

# Run tests
npm run test

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage
```

## Component Guide

### Pages

#### Configure.tsx
Create new discovery sessions with form validation and dataset upload.

**Features:**
- Research objective input
- Dataset upload
- Parameter configuration (max cycles, budget)
- Form validation
- Start discovery

#### Dashboard.tsx
Real-time monitoring of active discoveries.

**Features:**
- Live metrics (cost, cycles, findings, hypotheses)
- WebSocket connection status
- Cost and cycle charts
- Task breakdown pie chart
- Live activity feed
- Stop discovery button

#### Explorer.tsx
Browse and filter discovery results.

**Features:**
- Findings list with confidence filtering
- Hypotheses with status indicators
- Papers list
- Responsive cards
- Loading states

#### WorldModelView.tsx
Interactive knowledge graph visualization.

**Features:**
- Cytoscape.js graph rendering
- Multiple layout algorithms
- Node/edge type filtering
- Interactive node selection
- Detail panel
- Legend
- Zoom and pan controls

#### Reports.tsx
View and export generated reports.

**Features:**
- Report list sidebar
- Markdown rendering
- Syntax highlighting
- Download as .md
- Copy to clipboard

### Components

#### Visualization

**GraphVisualization.tsx**
- Main Cytoscape.js graph renderer
- Handles node/edge filtering
- Layout management
- Node selection events

**GraphControls.tsx**
- Layout selector (cose, circle, grid, etc.)
- Zoom controls
- Node/edge type filters

**NodeDetailPanel.tsx**
- Displays selected node information
- Shows metadata, confidence
- Collapsible design

**LegendPanel.tsx**
- Shows node types with colors and shapes
- Shows edge types with line styles
- Usage instructions

#### Dashboard

**CostChart.tsx**
- Line chart showing cost over cycles
- Per-cycle and cumulative costs
- Recharts implementation

**CycleTimeline.tsx**
- Bar chart of cycle performance
- Duration, tasks, findings per cycle

**TaskBreakdown.tsx**
- Pie chart of task type distribution
- Color-coded legend

#### Reports

**ReportList.tsx**
- Sidebar list of available reports
- Created timestamp
- Selection highlighting

**ReportViewer.tsx**
- React Markdown renderer
- Syntax highlighting with Prism
- Custom component styling
- Dark mode support

**ReportActions.tsx**
- Download button
- Copy to clipboard button
- Toast notifications

#### Common

**ErrorBoundary.tsx**
- Catches React errors
- Displays friendly error message
- Try again / go home buttons

**Skeleton.tsx**
- Loading placeholder components
- CardSkeleton, TableSkeleton, ListSkeleton
- GraphSkeleton, ChartSkeleton

## Hooks

### useDiscovery(discoveryId)
Fetch discovery status and metrics.

```typescript
const { data: status, isLoading } = useDiscovery(discoveryId)
const { data: metrics } = useDiscoveryMetrics(discoveryId)
```

### useWorldModel
Query world model data.

```typescript
const { data: graphData } = useWorldModelGraph(discoveryId)
const { data: findings } = useFindings(discoveryId, 0.7)
const { data: hypotheses } = useHypotheses(discoveryId)
```

### useWebSocket(discoveryId)
Real-time event streaming.

```typescript
const { messages, isConnected } = useWebSocket(discoveryId)
```

## Styling

Using Tailwind CSS with custom configuration:

- Dark mode support (`dark:` variants)
- Custom color palette
- Responsive breakpoints
- Utility-first approach

### Dark Mode
Automatically uses system preference. Toggle support can be added later.

### Common Patterns

```tsx
// Card with dark mode
<div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600">
  <h2 className="text-gray-900 dark:text-white">Title</h2>
  <p className="text-gray-600 dark:text-gray-400">Content</p>
</div>

// Button
<button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md">
  Click me
</button>
```

## API Integration

All API calls go through Axios instance in `src/services/api.ts`.

Base URL configured via environment variable `VITE_API_URL`.

### Example Service

```typescript
import api from './api'

export const exampleApi = {
  async getData(id: string) {
    const { data } = await api.get(`/endpoint/${id}`)
    return data
  }
}
```

### React Query Usage

```typescript
import { useQuery } from '@tanstack/react-query'
import { exampleApi } from '@/services/exampleApi'

const { data, isLoading, error } = useQuery({
  queryKey: ['example', id],
  queryFn: () => exampleApi.getData(id),
  enabled: !!id
})
```

## Testing

Using Vitest + React Testing Library.

```bash
# Run tests
npm run test

# Run with UI
npm run test:ui

# Coverage report
npm run test:coverage
```

### Example Test

```typescript
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import MyComponent from './MyComponent'

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent title="Test" />)
    expect(screen.getByText('Test')).toBeInTheDocument()
  })
})
```

## Build & Deployment

### Development Build

```bash
npm run dev
```

### Production Build

```bash
npm run build
# Output in dist/
```

### Docker Build

```bash
docker build -t kramer-frontend .
docker run -p 80:80 kramer-frontend
```

The Dockerfile uses multi-stage build:
1. Build stage: Compile TypeScript and bundle with Vite
2. Production stage: Serve with Nginx

## Environment Variables

Create `.env` file:

```bash
# API endpoints
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000

# Environment
VITE_NODE_ENV=development
```

**Note:** Vite requires `VITE_` prefix for environment variables.

## Troubleshooting

### API Connection Failed
- Ensure backend is running on port 8000
- Check `VITE_API_URL` in `.env`
- Check CORS settings in backend

### WebSocket Not Connecting
- Verify `VITE_WS_URL` in `.env`
- Check browser console for errors
- Ensure backend WebSocket endpoint is accessible

### Graph Not Rendering
- Check browser console for Cytoscape errors
- Verify graph data format from API
- Ensure container has dimensions

### Build Errors
```bash
# Clear cache
rm -rf node_modules dist .vite
npm install
npm run build
```

## Contributing

### Code Style
- Use TypeScript for all new files
- Follow existing component patterns
- Use Tailwind for styling
- Add dark mode support
- Write tests for new features

### Component Checklist
- [ ] TypeScript types defined
- [ ] Error handling
- [ ] Loading states
- [ ] Dark mode support
- [ ] Responsive design
- [ ] Tests written
- [ ] Documented

## License

See main project LICENSE file.
