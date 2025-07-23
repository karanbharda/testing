/**
 * WebSocket service for real-time trading bot updates
 * Handles connection management, reconnection, and data broadcasting
 */

class WebSocketService {
  constructor() {
    this.ws = null;
    this.url = process.env.NODE_ENV === 'production'
      ? 'ws://127.0.0.1:5000/ws'
      : 'ws://127.0.0.1:5000/ws';
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectInterval = 3000; // 3 seconds
    this.heartbeatInterval = 30000; // 30 seconds
    this.heartbeatTimer = null;
    this.reconnectTimer = null;
    this.isConnecting = false;
    this.isManualClose = false;

    // Event listeners
    this.listeners = {
      portfolio_update: [],
      trade_update: [],
      bot_status: [],
      initial_data: [],
      connection_status: [],
      error: []
    };
  }

  /**
   * Connect to WebSocket server
   */
  connect() {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      return Promise.resolve();
    }

    this.isConnecting = true;
    this.isManualClose = false;

    return new Promise((resolve, reject) => {
      try {
        console.log('Connecting to WebSocket:', this.url);
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('WebSocket connected successfully');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.emit('connection_status', { connected: true, reconnecting: false });

          // Request initial data
          this.send('get_initial_data');
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
            this.emit('error', { type: 'parse_error', error: error.message });
          }
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket connection closed:', event.code, event.reason);
          this.isConnecting = false;
          this.stopHeartbeat();
          this.emit('connection_status', { connected: false, reconnecting: !this.isManualClose });

          if (!this.isManualClose) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.isConnecting = false;
          this.emit('error', { type: 'connection_error', error: error.message || 'Connection failed' });
          reject(error);
        };

      } catch (error) {
        this.isConnecting = false;
        console.error('Error creating WebSocket connection:', error);
        this.emit('error', { type: 'creation_error', error: error.message });
        reject(error);
      }
    });
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect() {
    this.isManualClose = true;
    this.stopHeartbeat();
    this.clearReconnectTimer();

    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect');
      this.ws = null;
    }

    this.emit('connection_status', { connected: false, reconnecting: false });
  }

  /**
   * Send message to server
   */
  send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(typeof message === 'string' ? message : JSON.stringify(message));
      return true;
    } else {
      console.warn('WebSocket not connected, cannot send message:', message);
      return false;
    }
  }

  /**
   * Handle incoming messages
   */
  handleMessage(data) {
    const { type, data: payload, timestamp } = data;

    console.log('Received WebSocket message:', type, timestamp);

    switch (type) {
      case 'portfolio_update':
        this.emit('portfolio_update', payload);
        break;
      case 'trade_update':
        this.emit('trade_update', payload);
        break;
      case 'bot_status':
        this.emit('bot_status', payload);
        break;
      case 'initial_data':
        this.emit('initial_data', payload);
        break;
      default:
        console.log('Unknown message type:', type, payload);
    }
  }

  /**
   * Start heartbeat to keep connection alive
   */
  startHeartbeat() {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send('ping');
      }
    }, this.heartbeatInterval);
  }

  /**
   * Stop heartbeat
   */
  stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Schedule reconnection attempt
   */
  scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('error', {
        type: 'max_reconnect_attempts',
        error: 'Failed to reconnect after maximum attempts'
      });
      return;
    }

    this.clearReconnectTimer();

    const delay = Math.min(this.reconnectInterval * Math.pow(2, this.reconnectAttempts), 30000);
    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts + 1} in ${delay}ms`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      this.emit('connection_status', {
        connected: false,
        reconnecting: true,
        attempt: this.reconnectAttempts
      });
      this.connect().catch(error => {
        console.error('Reconnection failed:', error);
      });
    }, delay);
  }

  /**
   * Clear reconnection timer
   */
  clearReconnectTimer() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  /**
   * Add event listener
   */
  on(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event].push(callback);
    } else {
      console.warn('Unknown event type:', event);
    }
  }

  /**
   * Remove event listener
   */
  off(event, callback) {
    if (this.listeners[event]) {
      const index = this.listeners[event].indexOf(callback);
      if (index > -1) {
        this.listeners[event].splice(index, 1);
      }
    }
  }

  /**
   * Emit event to all listeners
   */
  emit(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Error in event listener:', error);
        }
      });
    }
  }

  /**
   * Get connection status
   */
  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get connection state
   */
  getConnectionState() {
    if (!this.ws) return 'DISCONNECTED';

    switch (this.ws.readyState) {
      case WebSocket.CONNECTING: return 'CONNECTING';
      case WebSocket.OPEN: return 'CONNECTED';
      case WebSocket.CLOSING: return 'CLOSING';
      case WebSocket.CLOSED: return 'DISCONNECTED';
      default: return 'UNKNOWN';
    }
  }
}

// Create singleton instance
const websocketService = new WebSocketService();

export default websocketService;
