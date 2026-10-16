"""Keep local service traffic on loopback even in a shell with proxy variables."""
from urllib.parse import urlparse


def is_loopback(url: str) -> bool:
    return urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"}


def mcp_transport(url: str):
    from fastmcp.client.transports import StreamableHttpTransport
    if not is_loopback(url):
        return StreamableHttpTransport(url)
    # FastMCP 4 / MCP 2 use httpx2 for their transport client.
    import httpx2

    def local_client(**kwargs):
        kwargs.setdefault("timeout", httpx2.Timeout(30, read=180))
        return httpx2.AsyncClient(trust_env=False, **kwargs)

    return StreamableHttpTransport(url, httpx_client_factory=local_client)
