# Kramer Web UI - Installation & Usage Guide

## Overview

The Kramer Web UI provides a modern, interactive interface for the Kramer autonomous discovery system, featuring:

- **Real-time monitoring** of discovery processes via WebSockets
- **Interactive dashboard** with live metrics and cost tracking
- **World model visualization** of the knowledge graph
- **Results explorer** for findings, hypotheses, and papers
- **Report viewer** for generated research reports

## Architecture

```
┌─────────────────────────────────────────┐
│          React Frontend (Port 3000)      │
│  Dashboard, Explorer, Visualization     │
└─────────────────────────────────────────┘
                    ↕ HTTP/WebSocket
┌─────────────────────────────────────────┐
│       FastAPI Backend (Port 8000)        │
│   REST API + WebSocket Event Streaming  │
└─────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│      Existing Kramer Components         │
│  CycleManager, WorldModel, Agents       │
└─────────────────────────────────────────┘
```

## Quick Start with Docker (Recommended)

### Prerequisites

- Docker and Docker Compose installed
- Anthropic API key

### Setup

1. **Set your API key** in `.env`:
```bash
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

2. **Start the application**:
```bash
docker-compose up --build
```

3. **Access the UI**:
- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs

### Stop the application:
```bash
docker-compose down
```

## Manual Setup (Development)

### Backend Setup

1. **Navigate to backend directory**:
```bash
cd backend
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set environment variables**:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

4. **Run the backend**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

### Frontend Setup

1. **Navigate to frontend directory**:
```bash
cd frontend
```

2. **Install dependencies**:
```bash
npm install
```

3. **Run the development server**:
```bash
npm run dev
```

The frontend will be available at http://localhost:3000

## Using the Web UI

### 1. Create a New Discovery

1. Navigate to the "New Discovery" page
2. Enter your research objective (e.g., "Investigate protein folding mechanisms")
3. Configure parameters:
   - **Max Cycles**: Number of discovery cycles (default: 20)
   - **Budget**: Maximum cost in USD (default: $100)
   - **Max Parallel Tasks**: Concurrent task execution (default: 4)
4. Click "Start Discovery"

### 2. Monitor Progress

Once started, you'll be redirected to the **Dashboard** where you can:

- View real-time metrics (cost, cycles, findings, hypotheses)
- Monitor live activity feed via WebSocket
- Track budget consumption
- Stop the discovery if needed

### 3. Explore Results

Navigate to the **Explorer** to view:

- **Findings**: Discovered scientific findings with confidence scores
- **Hypotheses**: Generated and tested hypotheses
- **Papers**: Retrieved research papers

### 4. Visualize Knowledge Graph

The **World Model** page shows:
- Interactive graph visualization of the knowledge network
- Node relationships (supports, refutes, derives from)
- Filter by node type (findings, hypotheses, papers)

### 5. View Reports

Access generated reports in the **Reports** section:
- Markdown-formatted research reports
- Export and download options

## API Endpoints

### Discovery Management

- `POST /api/v1/discovery/start` - Start new discovery
- `GET /api/v1/discovery/{id}/status` - Get status
- `POST /api/v1/discovery/{id}/stop` - Stop discovery
- `GET /api/v1/discovery/{id}/cycles` - Get cycle history
- `GET /api/v1/discovery/{id}/metrics` - Get metrics

### World Model

- `GET /api/v1/world-model/{id}/graph` - Get graph data
- `GET /api/v1/world-model/{id}/findings` - Get findings
- `GET /api/v1/world-model/{id}/hypotheses` - Get hypotheses
- `GET /api/v1/world-model/{id}/papers` - Get papers

### WebSocket

- `WS /api/v1/ws/{discovery_id}` - Real-time event stream

See full API documentation at http://localhost:8000/docs

## Configuration

### Backend Configuration

Edit `backend/app/config.py`:

```python
class Settings(BaseSettings):
    # API Settings
    API_V1_PREFIX: str = "/api/v1"

    # CORS Origins
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # Data Paths
    DATA_DIR: str = "../data"
    OUTPUTS_DIR: str = "../outputs"

    # Database
    DATABASE_PATH: str = "../data/discoveries.db"
```

### Frontend Configuration

Create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## Development

### Backend Development

Run tests:
```bash
cd backend
pytest tests/
```

Add new endpoint:
1. Create model in `app/models/`
2. Add service in `app/services/`
3. Create endpoint in `app/api/v1/`
4. Register in `app/api/v1/router.py`

### Frontend Development

Run tests:
```bash
cd frontend
npm test
```

Build for production:
```bash
npm run build
```

Add new page:
1. Create component in `src/pages/`
2. Add route in `src/App.tsx`
3. Add navigation in `src/components/common/Sidebar.tsx`

## Troubleshooting

### Backend Issues

**API won't start:**
- Check that port 8000 is available
- Verify Python dependencies are installed
- Check ANTHROPIC_API_KEY is set

**WebSocket disconnects:**
- Check firewall settings
- Verify WebSocket support in proxy/nginx

### Frontend Issues

**Build fails:**
- Run `npm install` to ensure dependencies
- Check Node.js version (requires 18+)

**API calls fail:**
- Verify backend is running on port 8000
- Check CORS settings in backend config
- Inspect browser console for errors

**WebSocket not connecting:**
- Verify backend WebSocket endpoint is accessible
- Check proxy configuration for WebSocket upgrade

## Production Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Manual Deployment

**Backend:**
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Frontend:**
```bash
npm run build
# Serve dist/ with nginx or similar
```

## Performance Optimization

### Backend

- Use multiple workers with Gunicorn
- Enable response caching for expensive queries
- Use connection pooling for database
- Implement rate limiting

### Frontend

- Enable production build optimizations
- Use code splitting for large components
- Implement virtual scrolling for large lists
- Cache API responses with React Query

## Security Considerations

- Set strong CORS policies in production
- Use HTTPS for all connections
- Implement authentication/authorization
- Sanitize user inputs
- Rate limit API endpoints
- Validate WebSocket messages

## Advanced Features (Future)

- [ ] Multi-user support with authentication
- [ ] Saved discovery templates
- [ ] Comparison of multiple discoveries
- [ ] Export results to various formats
- [ ] Advanced graph filtering and search
- [ ] Real-time collaboration features
- [ ] Mobile responsive design improvements

## Contributing

See `CONTRIBUTING.md` for guidelines on:
- Code style and conventions
- Pull request process
- Testing requirements
- Documentation standards

## Support

For issues and questions:
- Check existing issues on GitHub
- Review documentation at /docs
- Contact the development team

## License

See LICENSE file for details.
