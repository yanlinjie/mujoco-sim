# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Read saved routes from the configured A* MCP service."""

from typing import Any

import httpx
from anyio import fail_after
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .http_client import ServiceError, validate_endpoint


class PlanningClient:
    """Query A* by MCP as needed, so pose queries can run independently."""

    def __init__(self, url: str) -> None:
        validate_endpoint(url)
        self.url = url

    async def read(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a read-only A* tool and require a structured response."""
        try:
            with fail_after(5):
                async with httpx.AsyncClient(trust_env=False, timeout=5) as http:
                    async with streamable_http_client(self.url, http_client=http) as (
                        read,
                        write,
                        _,
                    ):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            result = await session.call_tool(tool, arguments)
        except Exception as exc:
            while getattr(exc, "exceptions", ()):
                exc = exc.exceptions[0]
            raise ServiceError(
                f"A* MCP {tool} failed: {exc or type(exc).__name__}"
            ) from exc
        if result.isError:
            detail = next(
                (part.text for part in result.content if part.type == "text"), ""
            )
            raise ServiceError(f"A* MCP {tool} failed: {detail}")
        data = result.structuredContent
        if not isinstance(data, dict) or data.get("ok") is not True:
            raise ServiceError(f"A* MCP {tool} returned an invalid response")
        return data
