"""
AutoPR MCP Client

Connects AutoPR to the local MCP server over stdio.
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PATH = (
    Path(__file__).resolve().parent
    / "mcp_server.py"
)


async def call_mcp_tool(
    tool_name: str,
    arguments: dict,
):
    """
    Start the local AutoPR MCP server, call one tool,
    and return the result.
    """

    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        env=None,
    )

    async with stdio_client(
        server_parameters
    ) as (read_stream, write_stream):

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:

            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments,
            )

            return result


def search_knowledge_via_mcp(
    query: str,
    top_k: int = 5,
):
    """
    Synchronous wrapper used by AutoPR.
    """

    return asyncio.run(
        call_mcp_tool(
            "search_knowledge_mcp",
            {
                "query": query,
                "top_k": top_k,
            },
        )
    )


if __name__ == "__main__":

    print("=== AUTOPR MCP CLIENT TEST ===")

    result = search_knowledge_via_mcp(
        "What are the security rules for API keys?",
        3,
    )

    print()
    print("MCP RESULT:")
    print(result)