#!/usr/bin/env python3
"""
Web Backend Entry Point
=======================

Standalone entry point for the web backend service.
Run this file to start the web backend independently.

Usage:
    python backend/web_backend_main.py
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("WEB_BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_BACKEND_PORT", "5000"))
    reload = os.getenv("WEB_BACKEND_RELOAD", "false").lower() == "true"
    
    print("=" * 60)
    print("Starting Web Backend Service")
    print("=" * 60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print("=" * 60)
    
    # Import and run the FastAPI app
    from web_backend import app
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

