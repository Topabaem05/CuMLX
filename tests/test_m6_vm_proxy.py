"""spikes/m6_vm_proxy.py on this machine's loopback, against a small server here. It makes no request to the internet."""

import importlib.util
from pathlib import Path
import socket
import threading

import pytest

SPEC = importlib.util.spec_from_file_location("m6_vm_proxy", Path(__file__).resolve().parents[1] / "spikes" / "m6_vm_proxy.py")
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)


@pytest.fixture
def upstream():
    """A server that records what it reads and answers once."""
    server = socket.create_server(("127.0.0.1", 0))
    seen = []

    def answer():
        conn, _ = server.accept()
        with conn:
            seen.append(conn.recv(4096))
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")

    threading.Thread(target=answer, daemon=True).start()
    yield server.getsockname()[1], seen
    server.close()


def ask(request):
    """Run one request through serve() and return all that the client reads."""
    client, inside = socket.socketpair()
    thread = threading.Thread(target=proxy.serve, args=(inside,), daemon=True)
    thread.start()
    client.sendall(request)
    return client, thread


def read_all(client):
    client.settimeout(5)
    got = b""
    while chunk := client.recv(4096):
        got += chunk
    return got


@pytest.mark.parametrize(
    "request_bytes",
    [
        b"CONNECT example.com:8080 HTTP/1.1\r\n\r\n",
        b"CONNECT example.com:22 HTTP/1.1\r\n\r\n",
        b"GET http://example.com:8080/ HTTP/1.1\r\n\r\n",
        b"GET /relative HTTP/1.1\r\n\r\n",
        b"nonsense\r\n\r\n",
    ],
)
def test_other_ports_and_bad_requests_are_refused(request_bytes):
    client, thread = ask(request_bytes)
    assert read_all(client).startswith(b"HTTP/1.1 403")
    thread.join(5)


def test_a_plain_request_reaches_the_server_in_origin_form(monkeypatch, upstream):
    port, seen = upstream
    monkeypatch.setattr(proxy, "ALLOWED_PORTS", {port})
    client, thread = ask(f"GET http://127.0.0.1:{port}/a?b=1 HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    assert read_all(client).endswith(b"ok")
    assert seen[0].startswith(b"GET /a?b=1 HTTP/1.1\r\nHost: x\r\n")
    thread.join(5)


def test_connect_opens_a_tunnel(monkeypatch, upstream):
    port, seen = upstream
    monkeypatch.setattr(proxy, "ALLOWED_PORTS", {port})
    client, thread = ask(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
    client.settimeout(5)
    assert client.recv(4096).startswith(b"HTTP/1.1 200")
    client.sendall(b"through")
    assert read_all(client).endswith(b"ok")
    assert seen == [b"through"]
    thread.join(5)


def test_an_unreachable_server_is_a_502(monkeypatch):
    closed = socket.create_server(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()
    monkeypatch.setattr(proxy, "ALLOWED_PORTS", {port})
    client, thread = ask(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
    assert read_all(client).startswith(b"HTTP/1.1 502")
    thread.join(5)


def test_it_listens_only_on_loopback():
    assert 'create_server(("127.0.0.1"' in Path(SPEC.origin).read_text()
