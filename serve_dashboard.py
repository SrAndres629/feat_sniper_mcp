"""
Dashboard HTTP Server
=====================
Serves dashboard.html on http://localhost:8080 to fix file:// origin issues
that block Supabase Realtime WebSocket connections.

Usage: python serve_dashboard.py
"""

import http.server
import socketserver
import webbrowser
import os

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/dashboard.html"
        print(f"🌐 Dashboard Server running at: {url}")
        print("   Press Ctrl+C to stop")
        webbrowser.open(url)
        httpd.serve_forever()
