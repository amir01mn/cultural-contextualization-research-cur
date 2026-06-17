from src.retriever import TripleRetriever, retrieve_triples


def test_retriever_returns_relevant_triple(sample_triples):
    retriever = TripleRetriever(sample_triples)
    hits = retriever.retrieve("What should a guest do with their shoes?", top_k=2)
    assert len(hits) >= 1
    top_triple, score = hits[0]
    assert "shoes" in (top_triple.subject + top_triple.object).lower()


def test_retriever_respects_top_k(sample_triples):
    hits = TripleRetriever(sample_triples).retrieve("respect", top_k=1)
    assert len(hits) == 1


def test_retriever_empty_triples():
    retriever = TripleRetriever([])
    assert retriever.retrieve("anything", top_k=5) == []


def test_convenience_function_returns_triples(sample_triples):
    result = retrieve_triples("removing shoes", sample_triples, top_k=2)
    assert all(hasattr(t, "triple_id") for t in result)
    assert len(result) <= 2


def test_backend_is_set(sample_triples):
    retriever = TripleRetriever(sample_triples)
    assert retriever.backend in {"tfidf", "sentence-transformers"}
