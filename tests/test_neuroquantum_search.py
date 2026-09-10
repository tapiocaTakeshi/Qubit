"""NeuroQuantum search: BM25 tokenisation/ranking, hybrid dense scoring, RAG prompts, persistence."""

import json

import pytest

from neuroquantum_search import (
    BM25Index,
    NeuroQuantumSearchIndex,
    build_rag_prompt,
    load_documents_from_file,
    tokenize_for_search,
)

DOCS = [
    "量子コンピュータは量子力学の原理を利用した次世代のコンピュータです。",
    "ニューラルネットワークは人間の脳の神経細胞の働きを模倣した計算モデルです。",
    "Python is a popular programming language for data science.",
]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokenizer_mixes_words_characters_and_bigrams():
    tokens = tokenize_for_search("量子力学 GPT-4!")
    assert "gpt" in tokens and "4" in tokens
    assert "量子" in tokens and "子力" in tokens and "力学" in tokens
    assert "!" not in tokens and "-" not in tokens


def test_tokenizer_normalises_width_and_case():
    assert tokenize_for_search("ＰＹＴＨＯＮ") == tokenize_for_search("python")
    assert tokenize_for_search("") == []
    assert tokenize_for_search("、。！") == []


def test_bigrams_do_not_cross_word_boundaries():
    tokens = tokenize_for_search("量子 AI 力学")
    assert "子力" not in tokens


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_bm25_ranks_matching_document_first_and_ignores_unknown_terms():
    bm25 = BM25Index()
    for d in DOCS:
        bm25.add(d)
    scores = bm25.scores("量子力学")
    assert scores.index(max(scores)) == 0
    assert scores[2] == 0.0
    assert bm25.scores("zzzz") == [0.0, 0.0, 0.0]
    assert bm25.scores("") == [0.0, 0.0, 0.0]


def test_bm25_empty_index_returns_empty():
    assert BM25Index().scores("量子") == []


# ---------------------------------------------------------------------------
# Index (BM25 only)
# ---------------------------------------------------------------------------


def test_index_search_returns_ranked_results_with_ids_and_metadata():
    index = NeuroQuantumSearchIndex()
    ids = index.add_documents(DOCS, ["q", None, "py"], [{"topic": "quantum"}, {}, {"topic": "code"}])
    assert ids == ["q", "doc-2", "py"]
    assert len(index) == 3 and index.mode == "bm25"

    hits = index.search("量子力学", top_k=5)
    assert [h.doc_id for h in hits] == ["q"]
    assert hits[0].rank == 1 and hits[0].score == 1.0 and hits[0].dense_score == 0.0
    assert hits[0].metadata == {"topic": "quantum"}

    hits = index.search("programming", top_k=1)
    assert hits[0].doc_id == "py"


def test_index_filters_and_thresholds():
    index = NeuroQuantumSearchIndex()
    index.add_documents(DOCS, ["q", "nn", "py"], [{"topic": "quantum"}, {"topic": "ai"}, {"topic": "code"}])
    assert [h.doc_id for h in index.search("コンピュータ モデル", metadata_filter={"topic": "ai"})] == ["nn"]
    assert index.search("コンピュータ", min_score=1.01) == []
    assert index.search("   ") == []
    assert index.search("量子", top_k=0) == []


def test_index_replace_remove_and_clear():
    index = NeuroQuantumSearchIndex()
    index.add_documents(DOCS, ["q", "nn", "py"])
    index.add_document("量子もつれについての新しい説明", doc_id="q")
    assert len(index) == 3
    assert index.get("q").text.startswith("量子もつれ")
    assert index.search("次世代") == []  # old text is gone

    assert index.remove_document("nn") is True
    assert index.remove_document("nn") is False
    assert len(index) == 2 and index.get("nn") is None
    assert [h.doc_id for h in index.search("python")] == ["py"]

    index.clear()
    assert len(index) == 0 and index.search("量子") == []


def test_index_ignores_blank_documents_and_validates_lengths():
    index = NeuroQuantumSearchIndex()
    assert index.add_documents(["", "   ", "本文"]) == ["doc-1"]
    with pytest.raises(ValueError):
        index.add_documents(["a"], ["x", "y"])
    with pytest.raises(ValueError):
        NeuroQuantumSearchIndex(alpha=1.5)


def test_dense_mode_requires_model():
    index = NeuroQuantumSearchIndex()
    index.add_documents(DOCS)
    with pytest.raises(ValueError):
        index.search("量子", mode="dense")
    with pytest.raises(ValueError):
        index.search("量子", mode="bogus")
    # hybrid silently degrades to bm25 without a model
    assert index.search("量子", mode="hybrid")[0].doc_id == "doc-1"


def test_save_and_load_roundtrip(tmp_path):
    index = NeuroQuantumSearchIndex(alpha=0.3)
    index.add_documents(DOCS, ["q", "nn", "py"], [{"topic": "quantum"}, {}, {}])
    path = tmp_path / "idx" / "index.json"
    index.save(str(path))

    restored = NeuroQuantumSearchIndex.from_file(str(path))
    assert len(restored) == 3 and restored.alpha == 0.3
    assert restored.get("q").metadata == {"topic": "quantum"}
    assert [h.doc_id for h in restored.search("量子力学")] == ["q"]


# ---------------------------------------------------------------------------
# RAG prompt + file loading
# ---------------------------------------------------------------------------


def test_build_rag_prompt_numbers_context_and_truncates():
    index = NeuroQuantumSearchIndex()
    index.add_documents(DOCS)
    hits = index.search("量子", top_k=2)
    prompt = build_rag_prompt("量子とは？", hits)
    assert prompt.startswith("以下の参考情報")
    assert "[1] 量子コンピュータ" in prompt and prompt.endswith("質問: 量子とは？\n回答:")

    short = build_rag_prompt("q", hits, max_context_chars=10)
    assert "…" in short
    assert build_rag_prompt("そのまま", []) == "そのまま"
    custom = build_rag_prompt("Q", hits, template="{context}|{query}", numbered=False)
    assert custom.startswith("量子コンピュータ") and custom.endswith("|Q")


def test_load_documents_from_text_json_and_jsonl(tmp_path):
    txt = tmp_path / "docs.txt"
    txt.write_text("段落一\n続き\n\n段落二\n", encoding="utf-8")
    texts, ids, metas = load_documents_from_file(str(txt))
    assert texts == ["段落一\n続き", "段落二"] and ids == [None, None]

    js = tmp_path / "docs.json"
    js.write_text(json.dumps(["a", {"text": "b", "id": "B", "metadata": {"k": 1}}]), encoding="utf-8")
    texts, ids, metas = load_documents_from_file(str(js))
    assert texts == ["a", "b"] and ids == [None, "B"] and metas == [{}, {"k": 1}]

    jl = tmp_path / "docs.jsonl"
    jl.write_text('{"text": "x", "id": 7}\n\n{"text": "y"}\n', encoding="utf-8")
    texts, ids, metas = load_documents_from_file(str(jl))
    assert texts == ["x", "y"] and ids == ["7", None]


# ---------------------------------------------------------------------------
# Hybrid search with a tiny NeuroQuantum model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    torch = pytest.importorskip("torch")
    from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

    with torch.random.fork_rng():
        torch.manual_seed(7)
        tokenizer = NeuroQuantumTokenizer(vocab_size=500)
        tokenizer.build_vocab(DOCS, min_freq=1)
        config = NeuroQuantumConfig(
            vocab_size=len(tokenizer), embed_dim=16, hidden_dim=32, num_heads=2,
            num_layers=1, max_seq_len=64, dropout=0.0,
        )
        model = NeuroQuantum(config=config)
    return model, tokenizer


def test_hybrid_search_uses_model_embeddings(tiny_model):
    model, tokenizer = tiny_model
    index = NeuroQuantumSearchIndex(model, tokenizer)
    index.add_documents(DOCS, ["q", "nn", "py"])
    assert index.mode == "hybrid" and index.has_dense

    hits = index.search("量子力学", top_k=3)
    assert hits[0].doc_id == "q"
    assert all(0.0 <= h.dense_score <= 1.0 + 1e-6 for h in hits)
    assert all(h.dense_score > 0 for h in hits)  # every doc gets a dense score
    expected = 0.5 * 1.0 + 0.5 * hits[0].dense_score
    assert hits[0].score == pytest.approx(expected, abs=1e-5)

    dense_only = index.search("量子力学", top_k=3, mode="dense")
    assert len(dense_only) == 3
    assert [h.score for h in dense_only] == sorted((h.score for h in dense_only), reverse=True)
    assert dense_only[0].bm25_score == 0.0

    # a document identical to the query is the closest in embedding space
    index.add_document("量子力学", doc_id="exact")
    assert index.search("量子力学", top_k=1, mode="dense")[0].doc_id == "exact"


def test_encoder_is_deterministic_and_restores_train_mode(tiny_model):
    model, tokenizer = tiny_model
    index = NeuroQuantumSearchIndex(model, tokenizer)
    model.train()
    a = index._encoder.encode(["量子", "脳の神経細胞"])
    b = index._encoder.encode(["量子", "脳の神経細胞"])
    assert a == b
    assert model.training  # restored
    assert len(a[0]) == 16
    model.eval()


def test_saved_embeddings_are_reused_on_load(tiny_model, tmp_path):
    model, tokenizer = tiny_model
    index = NeuroQuantumSearchIndex(model, tokenizer)
    index.add_documents(DOCS, ["q", "nn", "py"])
    path = tmp_path / "hybrid.json"
    index.save(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["embeddings"]) == 3

    restored = NeuroQuantumSearchIndex(model, tokenizer)
    restored.load(str(path))
    assert restored.search("量子力学")[0].dense_score == index.search("量子力学")[0].dense_score

    bm25_only = NeuroQuantumSearchIndex.from_file(str(path))
    assert bm25_only.mode == "bm25" and len(bm25_only) == 3


def test_neuroquantum_ai_search_and_rag(tiny_model, monkeypatch):
    from neuroquantum_layered import NeuroQuantumAI

    model, tokenizer = tiny_model
    ai = NeuroQuantumAI.__new__(NeuroQuantumAI)  # skip heavy __init__
    ai.model, ai.tokenizer, ai.device = model, tokenizer, next(model.parameters()).device
    ai._search_index = None

    assert ai.add_documents(DOCS, ["q", "nn", "py"]) == ["q", "nn", "py"]
    assert ai.search("量子力学", top_k=1)[0].doc_id == "q"

    captured = {}

    def fake_generate(prompt="", **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return "生成結果"

    monkeypatch.setattr(ai, "generate", fake_generate)
    text, hits = ai.generate_with_search("量子力学とは？", top_k=2, return_results=True, max_length=5)
    assert text == "生成結果" and hits[0].doc_id == "q"
    assert "[1] 量子コンピュータ" in captured["prompt"] and captured["kwargs"] == {"max_length": 5}

    ai.clear_documents()
    assert ai.generate_with_search("量子") == "生成結果"
    assert captured["prompt"] == "量子"  # no context when the index is empty
