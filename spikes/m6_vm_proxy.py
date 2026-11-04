"""A small web proxy for a VM whose NAT does not reach the internet. It listens only on this Mac's loopback, and goes only to ports 80 and 443.

    python3 spikes/m6_vm_proxy.py PORT

spikes/m6_vm_reproduce.sh reaches it from the guest through `ssh -R`. It changes no setting on this Mac. Standard library only, so it runs with any python3.
"""

import contextlib
import socket
import sys
import threading
from urllib.parse import urlsplit

ALLOWED_PORTS = {80, 443}
MAX_HEADER = 65536


def pipe(source, sink):
    try:
        while chunk := source.recv(65536):
            sink.sendall(chunk)
    except OSError:
        pass
    finally:
        for sock in (source, sink):
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)


def read_head(client):
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = client.recv(4096)
        if not chunk or len(head) > MAX_HEADER:
            return None, b""
        head += chunk
    end = head.index(b"\r\n\r\n") + 4
    return head[:end], head[end:]


def target_of(head):
    """(host, port, bytes to send first), or None if this request is refused."""
    line, _, rest = head.partition(b"\r\n")
    parts = line.decode("latin-1").split(" ")
    if len(parts) != 3:
        return None
    method, target, version = parts
    if method == "CONNECT":
        host, _, port = target.rpartition(":")
        if not host or not port.isdigit():
            return None
        return host.strip("[]"), int(port), None
    url = urlsplit(target)
    if url.scheme != "http" or not url.hostname:
        return None
    path = url.path or "/"
    if url.query:
        path += "?" + url.query
    return url.hostname, url.port or 80, f"{method} {path} {version}\r\n".encode("latin-1") + rest


def serve(client):
    with client:
        head, extra = read_head(client)
        if head is None:
            return
        target = target_of(head)
        if target is None or target[1] not in ALLOWED_PORTS:
            client.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return
        host, port, first = target
        try:
            upstream = socket.create_connection((host, port), timeout=30)
        except OSError:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return
        with upstream:
            upstream.settimeout(None)
            if first is None:
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            else:
                upstream.sendall(first)
            if extra:
                upstream.sendall(extra)
            other = threading.Thread(target=pipe, args=(upstream, client), daemon=True)
            other.start()
            pipe(client, upstream)
            other.join()


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    listener = socket.create_server(("127.0.0.1", int(sys.argv[1])))
    print(f"listening on 127.0.0.1:{listener.getsockname()[1]}", flush=True)
    while True:
        client, _ = listener.accept()
        threading.Thread(target=serve, args=(client,), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
