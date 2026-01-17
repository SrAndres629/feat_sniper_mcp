"""
Dashboard HTTP Server [LEGACY/DEPRECATED]
=========================================
Note: Use `nexus.bat` to launch the modern Streamlit Visual Cortex.
This script remains as a fallback for the old dashboard.html.
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
