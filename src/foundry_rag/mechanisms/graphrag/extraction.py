"""LLM entity/relation extraction for GraphRAG."""

from __future__ import annotations

import json

from foundry_rag.chunking.fixed import FixedSizeChunker
from foundry_rag.llm import chat_complete, embed_text
from foundry_rag.mechanisms.graphrag.models import Entity, KnowledgeGraph, Relationship, merge_graphs
from foundry_rag.mechanisms.graphrag import prompts


def _parse_kg_json(raw: str) -> KnowledgeGraph:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    # Extract outermost JSON object if the model added prose.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Last-chance: replace trailing commas before } or ]
        import re

        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        payload = json.loads(cleaned)
    entities = [
        Entity(
            name=e["name"],
            type=e["type"],
            description=e["description"],
        )
        for e in payload.get("entities", [])
    ]
    relationships = [
        Relationship(
            source=r["source"],
            target=r["target"],
            label=r["label"],
            description=r["description"],
            weight=float(r.get("weight", 0.5)),
        )
        for r in payload.get("relationships", [])
    ]
    return KnowledgeGraph(entities=entities, relationships=relationships)


def extract_graph_from_chunk(chunk: str) -> KnowledgeGraph:
    user = (
        prompts.get_user_prompt(chunk)
        + "\n\nReturn ONLY valid JSON matching the schema. "
        "Keep the graph compact: at most 8 entities and 10 relationships."
    )
    last_error: Exception | None = None
    for _ in range(3):
        raw = chat_complete(
            system=prompts.get_system_prompt(),
            user=user,
            max_output_tokens=2500,
            json_mode=True,
        )
        try:
            graph = _parse_kg_json(raw)
            break
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            user = (
                prompts.get_user_prompt(chunk)
                + "\n\nYour previous reply was invalid JSON. Return ONLY a single "
                "JSON object with keys entities and relationships."
            )
    else:
        raise RuntimeError(f"Failed to parse knowledge graph JSON: {last_error}")

    for entity in graph.entities:
        entity.embedding = embed_text(f"{entity.name}: {entity.description}")
    return graph


def extract_graph_from_document(
    text: str,
    *,
    chunk_size: int = 512,
    overlap_percentage: float = 10.0,
    max_chunks: int | None = None,
) -> KnowledgeGraph:
    chunks = FixedSizeChunker().create_chunks(
        text, chunk_size=chunk_size, overlap_percentage=overlap_percentage
    )
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    graphs = [extract_graph_from_chunk(chunk) for chunk in chunks]
    return merge_graphs(graphs)
