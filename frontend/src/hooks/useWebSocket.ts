import { useEffect, useState, useRef } from 'react'

interface WebSocketMessage {
  type: string
  discovery_id: string
  timestamp: string
  data: any
}

export function useWebSocket(discoveryId: string) {
  const [messages, setMessages] = useState<WebSocketMessage[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!discoveryId) return

    const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'
    const ws = new WebSocket(`${WS_URL}/api/v1/ws/${discoveryId}`)

    ws.onopen = () => {
      console.log('WebSocket connected')
      setIsConnected(true)
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setIsConnected(false)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WebSocketMessage
        setMessages((prev) => [...prev, message])
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    wsRef.current = ws

    // Keep-alive ping every 30 seconds
    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping')
      }
    }, 30000)

    return () => {
      clearInterval(pingInterval)
      ws.close()
    }
  }, [discoveryId])

  const clearMessages = () => setMessages([])

  return {
    messages,
    isConnected,
    clearMessages,
  }
}
