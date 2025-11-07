#!/bin/bash

# Kramer Web UI Startup Script

set -e

echo "🚀 Starting Kramer Web UI..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "⚠️  Please edit .env and add your ANTHROPIC_API_KEY"
    echo "Then run this script again."
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

echo "✓ Docker is running"
echo "✓ .env file found"
echo ""

# Build and start services
echo "📦 Building containers (this may take a few minutes)..."
docker-compose build

echo ""
echo "🎯 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check if services are running
if docker-compose ps | grep -q "kramer-backend.*Up"; then
    echo "✓ Backend is running"
else
    echo "❌ Backend failed to start"
    echo "Check logs with: docker-compose logs backend"
    exit 1
fi

if docker-compose ps | grep -q "kramer-frontend.*Up"; then
    echo "✓ Frontend is running"
else
    echo "❌ Frontend failed to start"
    echo "Check logs with: docker-compose logs frontend"
    exit 1
fi

echo ""
echo "✅ Kramer Web UI is now running!"
echo ""
echo "📍 Access the application:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend API:  http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "📊 View logs:"
echo "   All:  docker-compose logs -f"
echo "   Backend:  docker-compose logs -f backend"
echo "   Frontend:  docker-compose logs -f frontend"
echo ""
echo "🛑 Stop the application:"
echo "   docker-compose down"
echo ""
