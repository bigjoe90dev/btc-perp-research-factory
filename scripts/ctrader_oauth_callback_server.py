import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

PORT = 5555

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        self.wfile.write(b"<h2>Callback received. You can close this tab.</h2>")
        self.wfile.write(f"<pre>{qs}</pre>".encode("utf-8"))

        print("\n=== cTrader OAuth Callback ===")
        print("PATH:", self.path)
        print("QUERY:", qs)
        print("=============================\n")

if __name__ == "__main__":
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Listening on http://127.0.0.1:{PORT}/callback")
        httpd.serve_forever()
