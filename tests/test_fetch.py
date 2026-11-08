import socket

import pytest

from baitnswitch import fetch
from baitnswitch.fetch import FetchError, check_url, extract_headlines


@pytest.fixture
def resolve_to(monkeypatch):
    def set_address(address):
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        monkeypatch.setattr(fetch.socket, "getaddrinfo",
                            lambda *a, **k: [(family, socket.SOCK_STREAM, 6, "", (address, 0))])
    return set_address


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1", "fd00::1"])
def test_rejects_private_addresses(resolve_to, address):
    resolve_to(address)
    with pytest.raises(FetchError, match="private"):
        check_url("http://innocent-looking.example.com/")


def test_allows_public_address(resolve_to):
    resolve_to("93.184.216.34")
    check_url("https://www.example.com/news")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "gopher://x", "http://example.com:8080/"])
def test_rejects_bad_schemes_and_ports(url):
    with pytest.raises(FetchError):
        check_url(url)


def test_redirects_are_rechecked(resolve_to, monkeypatch):
    class Redirect:
        is_redirect = True
        headers = {"Location": "http://metadata.internal/latest"}
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    addresses = iter(["93.184.216.34", "169.254.169.254"])
    monkeypatch.setattr(fetch.socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(addresses), 0))])
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: Redirect())
    with pytest.raises(FetchError, match="private"):
        fetch.fetch_html("https://www.example.com/")


def test_extract_headlines_dedupes_and_drops_short():
    html = """
      <h1>Senate passes budget bill after overnight session</h1>
      <h2>Menu</h2>
      <h3>You won't believe what happened next</h3>
      <h4>You won't believe what happened next</h4>
    """
    assert extract_headlines(html) == [
        "Senate passes budget bill after overnight session",
        "You won't believe what happened next",
    ]
