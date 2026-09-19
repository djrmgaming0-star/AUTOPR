"""
AutoPR MCP Server

MCP v2 server exposing AutoPR RAG capabilities.
"""

from mcp.server.mcpserver import MCPServer

from rag.retriever import search_knowledge


# ============================================================
# MCP SERVER
# ============================================================

mcp = MCPServer("AutoPR")


# ============================================================
# RAG TOOL
# ============================================================

@mcp.tool()
def search_knowledge_mcp(
    query: str,
    top_k: int = 5,
) -> list:
    """
    Search the AutoPR knowledge base using RAG.

    Returns relevant repository rules, coding standards,
    testing rules, architecture information, and previous
    PR patterns.
    """

    return search_knowledge(
        query=query,
        top_k=top_k,
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@mcp.tool()
def health_check() -> str:
    """
    Check whether the AutoPR MCP server is running.
    """

    return "AutoPR MCP server is running."


# ============================================================
# SERVER ENTRY POINT
# ============================================================

if __name__ == "__main__":
    mcp.run()