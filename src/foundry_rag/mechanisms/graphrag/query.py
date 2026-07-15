"""Graph query + answer generation for GraphRAG."""

from __future__ import annotations

from typing import Any

from foundry_rag.llm import chat_complete, embed_text
from foundry_rag.mechanisms.graphrag.graph_db import GraphDb
from foundry_rag.mechanisms.graphrag import prompts


def format_knowledge_graph_as_context(result: dict[str, Any]) -> str:
    entities = {e["name"]: e for e in result.get("seeds", [])}
    for e in result.get("traversed", []):
        entities.setdefault(e["name"], e)

    lines = ["## Entities"]
    for entity in entities.values():
        lines.append(
            f"- {entity.get('name')} ({entity.get('type')}): {entity.get('description')}"
        )
    lines.append("")
    lines.append("## Relationships")
    for rel in result.get("relationships", []):
        lines.append(
            f"- {rel.get('source')} --[{rel.get('label')}]--> {rel.get('target')} "
            f"(weight: {float(rel.get('weight') or 0.5):.2f}): {rel.get('description')}"
        )
    return "\n".join(lines)


def answer_question(
    question: str,
    *,
    top_k: int = 5,
    traversal_depth: int = 1,
    min_path_score: float = 0.5,
) -> tuple[str, dict[str, Any]]:
    query_vector = embed_text(question)
    with GraphDb() as db:
        result = db.get_top_entities(
            query_vector,
            top_k=top_k,
            traversal_depth=traversal_depth,
            min_path_score=min_path_score,
        )
    context = format_knowledge_graph_as_context(result)
    answer = chat_complete(
        system=prompts.get_rag_system_prompt(context),
        user=question,
        max_output_tokens=700,
    )
    return answer, result
