from foundry_rag.shared.vector_db import InMemoryVectorDb, VectorSearchRecord


def test_search_orders_by_cosine_similarity_descending():
    db = InMemoryVectorDb(supported_vector_dimension=4)

    db.index(VectorSearchRecord(id="a", vector=[1.0, 0.0, 0.0, 0.0], data={"label": "identical"}))
    db.index(VectorSearchRecord(id="b", vector=[0.9, 0.1, 0.0, 0.0], data={"label": "similar"}))
    db.index(VectorSearchRecord(id="c", vector=[0.0, 1.0, 0.0, 0.0], data={"label": "orthogonal"}))

    results = db.search(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=3)

    assert [result.id for result in results] == ["a", "b", "c"]
    assert results[0].similarity == 1.0
    assert results[1].similarity == 0.99
    assert results[2].similarity == 0.0


def test_search_top_k_limits_result_count():
    db = InMemoryVectorDb(supported_vector_dimension=4)

    db.index(VectorSearchRecord(id="a", vector=[1.0, 0.0, 0.0, 0.0]))
    db.index(VectorSearchRecord(id="b", vector=[0.9, 0.1, 0.0, 0.0]))
    db.index(VectorSearchRecord(id="c", vector=[0.0, 1.0, 0.0, 0.0]))

    results = db.search(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=1)

    assert [result.id for result in results] == ["a"]


def test_index_rejects_duplicate_id():
    db = InMemoryVectorDb(supported_vector_dimension=4)
    db.index(VectorSearchRecord(id="a", vector=[1.0, 0.0, 0.0, 0.0]))

    try:
        db.index(VectorSearchRecord(id="a", vector=[0.0, 1.0, 0.0, 0.0]))
        assert False, "expected ValueError for duplicate id"
    except ValueError:
        pass


def test_index_rejects_wrong_dimension():
    db = InMemoryVectorDb(supported_vector_dimension=4)

    try:
        db.index(VectorSearchRecord(id="a", vector=[1.0, 0.0]))
        assert False, "expected ValueError for wrong dimension"
    except ValueError:
        pass
