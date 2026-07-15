"""Neo4j wrapper for GraphRAG — port of `08_GraphRAG/GraphDb.cs`."""

from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from foundry_rag import config
from foundry_rag.mechanisms.graphrag.models import KnowledgeGraph


class GraphDb:
    VECTOR_INDEX_NAME = "entity_embeddings"
    EMBEDDING_DIMENSIONS = 1536

    def __init__(self) -> None:
        if not (config.NEO4J_URI and config.NEO4J_USER and config.NEO4J_PASSWORD):
            raise RuntimeError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are required")
        self._driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> GraphDb:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def ensure_vector_index(self) -> None:
        query = f"""
        CREATE VECTOR INDEX {self.VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (e:Entity)
        ON e.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.EMBEDDING_DIMENSIONS},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        with self._driver.session() as session:
            session.run(query)

    def ingest_graph(self, graph: KnowledgeGraph) -> None:
        with self._driver.session() as session:
            for entity in graph.entities:
                # Label from entity type — sanitize to Neo4j identifier
                label = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in entity.type)
                session.run(
                    f"""
                    MERGE (e:{label} {{name: $name}})
                    ON CREATE SET e:Entity, e.description = $description, e.embedding = $embedding
                    ON MATCH SET e:Entity, e.description = $description, e.embedding = $embedding
                    """,
                    name=entity.name,
                    description=entity.description,
                    embedding=entity.embedding,
                )
            for rel in graph.relationships:
                label = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in rel.label.upper())
                session.run(
                    f"""
                    MATCH (s {{name: $source}}), (t {{name: $target}})
                    MERGE (s)-[r:{label}]->(t)
                    ON CREATE SET r.description = $description, r.weight = $weight
                    ON MATCH  SET r.description = $description, r.weight = (r.weight + $weight) / 2.0
                    """,
                    source=rel.source,
                    target=rel.target,
                    description=rel.description,
                    weight=rel.weight,
                )

    def count_graph(self) -> tuple[int, int]:
        with self._driver.session() as session:
            nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return int(nodes), int(edges)

    def get_top_entities(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        traversal_depth: int = 1,
        min_path_score: float = 0.5,
    ) -> dict[str, Any]:
        with self._driver.session() as session:
            seed_rows = list(
                session.run(
                    f"""
                    CALL db.index.vector.queryNodes('{self.VECTOR_INDEX_NAME}', $topK, $queryVector)
                    YIELD node, score
                    RETURN node, score
                    """,
                    topK=top_k,
                    queryVector=query_vector,
                )
            )

        seed_entities = []
        traversed: dict[str, dict[str, Any]] = {}
        relationships: dict[tuple[str, str, str], dict[str, Any]] = {}

        with self._driver.session() as session:
            for row in seed_rows:
                node = row["node"]
                labels = [l for l in node.labels if l != "Entity"]
                seed_entities.append(
                    {
                        "name": node.get("name"),
                        "type": labels[0] if labels else "Entity",
                        "description": node.get("description"),
                    }
                )
                paths = list(
                    session.run(
                        f"""
                        MATCH (seed:Entity {{name: $name}})
                        OPTIONAL MATCH path = (seed)-[rels*1..{traversal_depth}]-(endNode)
                        WITH path,
                             CASE
                               WHEN path IS NULL THEN null
                               WHEN reduce(s = 1.0, r IN relationships(path) | s * coalesce(r.weight, 0.5)) >= $minPathScore
                               THEN path
                               ELSE null
                             END AS qualified
                        RETURN qualified AS path
                        """,
                        name=node.get("name"),
                        minPathScore=min_path_score,
                    )
                )
                for path_row in paths:
                    path = path_row["path"]
                    if path is None:
                        continue
                    for n in path.nodes:
                        name = n.get("name")
                        if not name:
                            continue
                        nlabels = [l for l in n.labels if l != "Entity"]
                        traversed[name] = {
                            "name": name,
                            "type": nlabels[0] if nlabels else "Entity",
                            "description": n.get("description"),
                        }
                    for rel in path.relationships:
                        key = (rel.start_node.get("name"), rel.end_node.get("name"), rel.type)
                        relationships[key] = {
                            "source": key[0],
                            "target": key[1],
                            "label": key[2],
                            "description": rel.get("description"),
                            "weight": rel.get("weight", 0.5),
                        }

        return {
            "seeds": seed_entities,
            "traversed": list(traversed.values()),
            "relationships": list(relationships.values()),
        }
