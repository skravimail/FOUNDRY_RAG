"""Create Azure AI Search indexes from reference index_definition.json files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from azure.search.documents.indexes.models import SearchIndex

from foundry_rag.clients import get_search_client, get_search_index_client
from foundry_rag.llm import embed_text
from foundry_rag.paths import data_file


def load_index_definition(module_folder: str) -> dict[str, Any]:
    path = data_file(module_folder, "index_definition.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _sanitize_index_definition(raw)


def _sanitize_index_definition(node: Any) -> Any:
    """Drop portal/export fields the current Search REST API rejects."""
    drop_keys = {
        "purviewEnabled",
        "flightingOptIn",
        # Some reference indexes embed a hardcoded Azure OpenAI vectorizer URI;
        # we pass VectorizedQuery vectors ourselves.
        "vectorizers",
    }
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key in drop_keys:
                continue
            cleaned[key] = _sanitize_index_definition(value)
        # If vectorizers were removed, also clear vectorizer references on profiles.
        vector_search = cleaned.get("vectorSearch")
        if isinstance(vector_search, dict):
            for profile in vector_search.get("profiles") or []:
                if isinstance(profile, dict):
                    profile.pop("vectorizer", None)
                    profile.pop("vectorizerName", None)
        return cleaned
    if isinstance(node, list):
        return [_sanitize_index_definition(item) for item in node]
    return node


def ensure_index(module_folder: str) -> SearchIndex:
    definition = load_index_definition(module_folder)
    client = get_search_index_client()
    index = SearchIndex(definition)
    return client.create_or_update_index(index)


def _flatten_starship(doc: dict[str, Any], *, include_notes_vector: bool) -> dict[str, Any]:
    specs = doc.get("Specifications") or {}
    overview = doc.get("Overview") or ""
    notes = doc.get("Notes") or ""
    payload: dict[str, Any] = {
        "Id": doc["Id"],
        "Title": doc.get("Title"),
        "Category": doc.get("Category"),
        "Overview": overview,
        "Features": doc.get("Features") or [],
        "OverviewVector": embed_text(overview),
    }
    # Module 02 index has optional structured fields (ProductId, specs, Notes).
    if "ProductId" in doc:
        payload["ProductId"] = doc.get("ProductId")
    if "TopSpeed" in specs or "Fuel" in specs:
        payload["TopSpeed"] = specs.get("TopSpeed")
        payload["Fuel"] = specs.get("Fuel")
        payload["Seats"] = specs.get("Seats")
        payload["ArtificialGravity"] = specs.get("ArtificialGravity")
        payload["Notes"] = notes
    if include_notes_vector:
        payload["NotesVector"] = embed_text(notes)
    return payload


def index_starships(module_folder: str) -> int:
    """Embed and upload starships.json into the module's index."""
    ensure_index(module_folder)
    definition = load_index_definition(module_folder)
    index_name = definition["name"]
    field_names = {f["name"] for f in definition["fields"]}
    include_notes_vector = "NotesVector" in field_names

    starships = json.loads(data_file(module_folder, "starships.json").read_text(encoding="utf-8"))
    docs = [
        _flatten_starship(ship, include_notes_vector=include_notes_vector) for ship in starships
    ]
    # Drop keys not in the index schema
    cleaned = []
    for doc in docs:
        cleaned.append({k: v for k, v in doc.items() if k in field_names})

    client = get_search_client(index_name)
    result = client.upload_documents(documents=cleaned)
    succeeded = sum(1 for r in result if r.succeeded)
    if succeeded != len(cleaned):
        failures = [r for r in result if not r.succeeded]
        raise RuntimeError(f"Document upload partially failed: {failures}")
    return succeeded
