"""Knowledge-graph models + merge aggregator for GraphRAG."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Entity:
    name: str
    type: str
    description: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class Relationship:
    source: str
    target: str
    label: str
    description: str
    weight: float


@dataclass
class KnowledgeGraph:
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)


def merge_graphs(graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
    """Entity first-wins; relationships dedupe on (source, target, label) with avg weight."""
    entities: dict[str, Entity] = {}
    relationships: dict[tuple[str, str, str], Relationship] = {}

    for graph in graphs:
        for entity in graph.entities:
            key = entity.name
            if key not in entities:
                entities[key] = entity
        for rel in graph.relationships:
            key = (rel.source, rel.target, rel.label.upper())
            existing = relationships.get(key)
            if existing is None:
                relationships[key] = Relationship(
                    source=rel.source,
                    target=rel.target,
                    label=rel.label.upper(),
                    description=rel.description,
                    weight=rel.weight,
                )
            else:
                existing.weight = (existing.weight + rel.weight) / 2.0
                if rel.description and not existing.description:
                    existing.description = rel.description

    return KnowledgeGraph(
        entities=list(entities.values()),
        relationships=list(relationships.values()),
    )
