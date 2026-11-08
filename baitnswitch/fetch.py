"""Fetch a public web page and pull out its headlines.

The API fetches URLs on behalf of callers, so it must not become a proxy into
private networks (SSRF). Every hop, including redirects, is checked: http(s)
only, default ports only, and the host must resolve exclusively to public IPs.
Remaining gap: a DNS answer can change between our check and the request
(rebinding). Closing that needs pinning the connection to the checked IP; for
a hobby API, the check plus short timeouts is a reasonable line.
"""

import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from .text import word_count

TIMEOUT = (3.05, 5)  # connect, read (seconds)
MAX_BYTES = 5_000_000
MAX_REDIRECTS = 3
USER_AGENT = "baitnswitch/2.0 (+https://github.com/SlimHintz/bait-n-switch)"
HEADLINE_TAGS = ["h1", "h2", "h3", "h4"]


class FetchError(ValueError):
    """The URL is not allowed or the page could not be retrieved."""


def check_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError("only http and https URLs are supported")
    if not parts.hostname:
        raise FetchError("URL has no host")
    if parts.port not in (None, 80, 443):
        raise FetchError("non-standard ports are not allowed")
    try:
        infos = socket.getaddrinfo(parts.hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchError(f"could not resolve {parts.hostname}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if not address.is_global:
            raise FetchError("URL points to a private or reserved address")


def fetch_html(url: str) -> str:
    for _ in range(MAX_REDIRECTS + 1):
        check_url(url)
        try:
            response = requests.get(url, timeout=TIMEOUT, allow_redirects=False, stream=True,
                                    headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
        except requests.RequestException as exc:
            raise FetchError("could not fetch the page") from exc
        with response:
            if response.is_redirect:
                url = urljoin(url, response.headers.get("Location", ""))
                continue
            if response.status_code != 200:
                raise FetchError(f"page returned HTTP {response.status_code}")
            if "html" not in response.headers.get("Content-Type", ""):
                raise FetchError("page is not HTML")
            body = b""
            for chunk in response.iter_content(65536):
                body += chunk
                if len(body) > MAX_BYTES:
                    raise FetchError("page is too large")
            return body.decode(response.encoding or "utf-8", errors="replace")
    raise FetchError("too many redirects")


def extract_headlines(html: str, min_words: int = 4) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    seen, headlines = set(), []
    for tag in soup.find_all(HEADLINE_TAGS):
        text = " ".join(tag.get_text(" ").split())
        if word_count(text) >= min_words and text not in seen:
            seen.add(text)
            headlines.append(text)
    return headlines
