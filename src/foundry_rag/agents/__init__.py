"""Foundry Agent Service ports of the RAG mechanisms.

Each module here recreates a mechanism from `foundry_rag.mechanisms` as a
server-side Foundry *prompt agent* (via the `azure-ai-projects` agents API),
using native Foundry tools (e.g. the Azure AI Search tool) instead of calling
the data plane directly.
"""
