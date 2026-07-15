"""Naive RAG: embed whole docs → in-memory cosine top-k → prompt stuffing.

Port of `01_NaiveRAG/NaiveRagExample.cs`.
"""

from __future__ import annotations

from foundry_rag.llm import chat_complete, embed_text
from foundry_rag.paths import data_file
from foundry_rag.shared.vector_db import InMemoryVectorDb, VectorSearchRecord, VectorSearchResult

DATA_SOURCE: list[tuple[str, str]] = [
    ("Aurora Class Shuttle", "aurora-class.md"),
    ("Ion‑Drive Clipper", "ion-drive-clipper.md"),
    ("Nebula-X Cruiser", "nebula-class.md"),
    ("Quantum-Fold Starliner", "quantum-fold-starliner.md"),
    ("Starlance Explorer", "starlance-explorer.md"),
]

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant for the Galactic Voyages travel agency.
Answer questions using only the information provided in the documents.
Keep your answers clear and friendly.
"""

KID_FRIENDLY_SYSTEM_PROMPT = """\
You explain things the way a kid would understand. 
Keep answers short, simple, and fun. 
Use playful analogies, like toys, snacks, or pets. 
"""

MARKETING_SYSTEM_PROMPT = """\
You are a friendly travel agent for Galactic Voyages.
Highlight the positive features of each ship.
Keep the tone upbeat and reassuring.
"""

SAMPLE_QUESTIONS = [
    "How fast is the Nebula-X Cruiser?",
    "What fuel does the Ion-Drive Clipper use?",
    "Does the Aurora-Class Shuttle have artificial gravity?",
    "Which ship can travel the farthest without refueling?",
]


def system_prompts() -> list[str]:
    return [DEFAULT_SYSTEM_PROMPT, KID_FRIENDLY_SYSTEM_PROMPT, MARKETING_SYSTEM_PROMPT]


def build_vector_db(module_folder: str = "01_naive_rag") -> InMemoryVectorDb:
    db = InMemoryVectorDb(supported_vector_dimension=1536)
    for ship_name, filename in DATA_SOURCE:
        text = data_file(module_folder, filename).read_text(encoding="utf-8")
        vector = embed_text(text)
        db.index(
            VectorSearchRecord(
                id=ship_name,
                vector=vector,
                data={"Text": text},
            )
        )
    return db


def create_user_prompt(question: str, results: list[VectorSearchResult]) -> str:
    lines = [
        "Here are the documents related to the question.",
        "Use only the information in these documents when answering.",
        "",
        "=== Retrieved Documents ===",
    ]
    for index, result in enumerate(results, start=1):
        lines.extend(["", f"[Document {index}]", result.data["Text"]])
    lines.extend(["", "=== User Question ===", question])
    return "\n".join(lines)


def answer_question(
    db: InMemoryVectorDb,
    question: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    top_k: int = 3,
) -> tuple[str, list[VectorSearchResult]]:
    query_vector = embed_text(question)
    results = db.search(query_vector, top_k)
    answer = chat_complete(
        system=system_prompt,
        user=create_user_prompt(question, results),
        max_output_tokens=600,
    )
    return answer, results
