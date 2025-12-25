#!/usr/bin/env python3
"""
Standalone HTTPS proxy to capture OAuth requests (no mitmproxy needed).

Usage:
1. Generate certificates (first time only):
   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
     -subj "/CN=localhost"

2. Run this script:
   python oauth_capture.py

3. In another terminal, run Claude Code with proxy and ignore cert errors:
   NODE_TLS_REJECT_UNAUTHORIZED=0 HTTPS_PROXY=http://127.0.0.1:8888 claude login

4. Complete the OAuth flow - requests will be logged to console

Alternative: Use curl to test directly (simpler):
   NODE_TLS_REJECT_UNAUTHORIZED=0 HTTPS_PROXY=http://127.0.0.1:8888 \
     curl -v https://console.anthropic.com/v1/oauth/token
"""

import http.server
import socketserver
import ssl
import socket
import threading
import json
import re

PORT = 8888

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_CONNECT(self):
        """Handle CONNECT for HTTPS tunneling."""
        host, port = self.path.split(':')
        port = int(port)

        print(f"\n[CONNECT] {host}:{port}")

        # Tell client we're ready
        self.send_response(200, 'Connection Established')
        self.end_headers()

        # Create connection to target
        target_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        target_sock.connect((host, port))

        # Wrap target with SSL
        context = ssl.create_default_context()
        target_ssl = context.wrap_socket(target_sock, server_hostname=host)

        # Now we need to intercept the client's request
        # Read from client, forward to server
        self._tunnel(target_ssl, host)

    def _tunnel(self, target_ssl, host):
        """Tunnel data between client and server, logging requests."""
        client_sock = self.connection

        # Read request from client
        data = b""
        while True:
            chunk = client_sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\r\n\r\n" in data:
                # Check if there's a Content-Length
                headers_end = data.index(b"\r\n\r\n")
                headers = data[:headers_end].decode('utf-8', errors='replace')
                match = re.search(r'Content-Length:\s*(\d+)', headers, re.IGNORECASE)
                if match:
                    content_length = int(match.group(1))
                    body_start = headers_end + 4
                    body_so_far = len(data) - body_start
                    remaining = content_length - body_so_far
                    while remaining > 0:
                        chunk = client_sock.recv(min(remaining, 4096))
                        if not chunk:
                            break
                        data += chunk
                        remaining -= len(chunk)
                break

        # Log the request
        if b"oauth/token" in data or b"console.anthropic.com" in host.encode():
            print("\n" + "=" * 70)
            print(f"[REQUEST to {host}]")
            print("=" * 70)
            try:
                decoded = data.decode('utf-8', errors='replace')
                # Split headers and body
                if "\r\n\r\n" in decoded:
                    headers, body = decoded.split("\r\n\r\n", 1)
                    print("HEADERS:")
                    for line in headers.split("\r\n"):
                        print(f"  {line}")
                    print("\nBODY:")
                    try:
                        print(json.dumps(json.loads(body), indent=2))
                    except:
                        print(body[:500])
                else:
                    print(decoded[:1000])
            except Exception as e:
                print(f"Error decoding: {e}")
            print("=" * 70)

        # Forward to target
        target_ssl.sendall(data)

        # Get response
        response = b""
        while True:
            try:
                chunk = target_ssl.recv(4096)
                if not chunk:
                    break
                response += chunk
            except:
                break

        # Log response
        if response:
            print("\n" + "-" * 70)
            print("[RESPONSE]")
            print("-" * 70)
            try:
                decoded = response.decode('utf-8', errors='replace')
                if "\r\n\r\n" in decoded:
                    headers, body = decoded.split("\r\n\r\n", 1)
                    print("STATUS + HEADERS:")
                    for line in headers.split("\r\n")[:10]:
                        print(f"  {line}")
                    print("\nBODY:")
                    try:
                        print(json.dumps(json.loads(body), indent=2))
                    except:
                        print(body[:500])
            except Exception as e:
                print(f"Error decoding response: {e}")
            print("-" * 70)

        # Forward response to client
        client_sock.sendall(response)

        target_ssl.close()

    def log_message(self, format, *args):
        pass  # Suppress default logging

if __name__ == "__main__":
    print(f"Starting proxy on port {PORT}")
    print("Run Claude Code with:")
    print(f"  NODE_TLS_REJECT_UNAUTHORIZED=0 HTTPS_PROXY=http://127.0.0.1:{PORT} claude login")
    print()

    with socketserver.TCPServer(("", PORT), ProxyHandler) as httpd:
        httpd.serve_forever()
