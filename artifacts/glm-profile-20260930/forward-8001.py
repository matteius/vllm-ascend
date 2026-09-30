"""Forward the Threadripper LAN port to its loopback GLM profiling server."""

import select
import socket
import socketserver

LISTEN_HOST = "192.168.53.187"
LISTEN_PORT = 8001
UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = 8003
BUFFER_BYTES = 65536


class RelayHandler(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.create_connection((UPSTREAM_HOST, UPSTREAM_PORT)) as upstream:
            peers = (self.request, upstream)
            while True:
                readable, _, _ = select.select(peers, (), ())
                for source in readable:
                    payload = source.recv(BUFFER_BYTES)
                    if not payload:
                        return
                    target = upstream if source is self.request else self.request
                    target.sendall(payload)


class ThreadedRelay(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with ThreadedRelay((LISTEN_HOST, LISTEN_PORT), RelayHandler) as server:
        print(f"Forwarding {LISTEN_HOST}:{LISTEN_PORT} to {UPSTREAM_HOST}:{UPSTREAM_PORT}", flush=True)
        server.serve_forever()
