\# AutoPR Architecture



AutoPR is an AI-powered software engineering agent.



The system receives a work item and uses tools to understand the

requirements, inspect the repository, modify code, run tests, and

prepare a pull request.



\## Main Components



1\. Work Item

&#x20;  - Contains the requested software change.



2\. Agent

&#x20;  - Understands the work item.

&#x20;  - Plans the implementation.

&#x20;  - Calls available tools.



3\. MCP Server

&#x20;  - Exposes repository and knowledge tools to the agent.



4\. Knowledge Base

&#x20;  - Contains repository rules, coding standards, testing rules,

&#x20;    architecture information, and previous implementation context.



5\. RAG Pipeline

&#x20;  - Retrieves relevant knowledge from the knowledge base.



6\. Repository

&#x20;  - Contains the source code being modified.



\## Intended Flow



Work Item

&#x20;   ↓

Agent

&#x20;   ↓

MCP Tools

&#x20;   ↓

Repository / Knowledge Base

&#x20;   ↓

Implementation

&#x20;   ↓

Tests

&#x20;   ↓

Pull Request

