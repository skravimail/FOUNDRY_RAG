"""Unit tests for T²-RAGBench Number Match / ranking metrics."""

from foundry_rag.eval.t2_ragbench.metrics import mrr_at_k, number_match, parse_floats, recall_at_k
from foundry_rag.eval.t2_ragbench.parse import context_id_from_blob_url, ranked_context_ids_from_result


def test_parse_floats_percent_and_commas():
    vals = parse_floats("about 9.3% change")
    assert any(abs(v - 0.093) < 1e-9 for v in vals)
    assert 1_234_567.0 in parse_floats("revenue was $1,234,567")
    assert 705.25 in parse_floats("average is 705.25 million")


def test_number_match_relative_eps():
    assert number_match("The answer is 704.25", "705.25")  # ~0.14% rel error
    assert not number_match("The answer is 600", "705.25")
    assert number_match("9.3%", "0.093")
    assert not number_match("no number here", "1.0")


def test_mrr_and_recall():
    ranked = ["a", "gold", "c"]
    assert mrr_at_k(ranked, "gold", k=3) == 0.5
    assert recall_at_k(ranked, "gold", k=3) == 1.0
    assert mrr_at_k(ranked, "missing", k=3) == 0.0
    assert recall_at_k(["x", "y", "z", "gold"], "gold", k=3) == 0.0


def test_context_id_from_blob_url():
    url = (
        "https://foudryragstorageacct.blob.core.windows.net/"
        "t2-ragbench/t2rag/FinQA/finqa_test_ctx_118/page_372.pdf"
    )
    assert context_id_from_blob_url(url) == "finqa_test_ctx_118"


def test_ranked_context_ids_dedupe():
    result = {
        "references": [
            {
                "blobUrl": "https://x/t2rag/FinQA/finqa_test_ctx_118/page_372.pdf",
                "rerankerScore": 3.0,
            },
            {
                "blobUrl": "https://x/t2rag/FinQA/finqa_test_ctx_118/page_372.pdf",
                "rerankerScore": 2.0,
            },
            {
                "blobUrl": "https://x/t2rag/FinQA/finqa_test_ctx_334/page_148.pdf",
                "rerankerScore": 1.0,
            },
        ]
    }
    assert ranked_context_ids_from_result(result, k=3) == [
        "finqa_test_ctx_118",
        "finqa_test_ctx_334",
    ]
