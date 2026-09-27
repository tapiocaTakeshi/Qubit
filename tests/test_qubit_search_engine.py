"""Qubit Search Engine: ranking, filters, incremental updates, recommendations, persistence and HTTP API."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from qubit_search_engine import SearchEngine, SearchError, SearchIndex, compile_filters, create_server

RIDDLES = [
    {"objectID": "r1", "title": "量子力学の謎", "tags": ["科学"], "category": "理系", "playCount": 5,
     "answer": "シュレディンガー", "createdAt": "2025-01-01T00:00:00Z"},
    {"objectID": "r2", "title": "ひらめき謎解き", "tags": ["初心者"], "category": "なぞなぞ", "playCount": 50,
     "createdAt": "2025-02-01T00:00:00Z"},
    {"objectID": "r3", "title": "Quantum puzzle", "description": "A quantum computing riddle", "playCount": 1,
     "contentType": "game"},
    {"objectID": "r4", "title": "謎解き脱出ゲーム", "tags": ["脱出", "初心者"], "category": "なぞなぞ", "playCount": 20},
]


@pytest.fixture
def engine():
    eng = SearchEngine()
    eng.batch("Riddles", [{"action": "addObject", "body": r} for r in RIDDLES])
    return eng


def ids(result):
    return [h["objectID"] for h in result["hits"]]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_japanese_query_matches_without_morphological_analysis(engine):
    assert ids(engine.search("Riddles", {"query": "量子"})) == ["r1"]
    # 「謎解き」を含む 2 件。同点は customRanking (playCount 降順) で並ぶ
    assert ids(engine.search("Riddles", {"query": "謎解き"}))[:2] == ["r2", "r4"]


def test_single_character_overlap_is_not_a_hit(engine):
    # 「学」の 1 文字しか共通しない文書は被覆率不足で除外
    assert ids(engine.search("Riddles", {"query": "学校"})) == []


def test_last_word_is_prefix_matched_while_typing(engine):
    assert ids(engine.search("Riddles", {"query": "quan"})) == ["r3"]
    assert ids(engine.search("Riddles", {"query": "quan", "prefixSearch": False})) == []


def test_title_outranks_body_and_width_is_normalised(engine):
    engine.save_object("Riddles", {"objectID": "r5", "title": "計算の話", "description": "quantum quantum"})
    assert ids(engine.search("Riddles", {"query": "ＱＵＡＮＴＵＭ"}))[0] == "r3"


def test_empty_query_returns_all_in_custom_ranking_order(engine):
    assert ids(engine.search("Riddles", {"query": ""})) == ["r2", "r4", "r1", "r3"]


def test_pagination_shape_matches_algolia(engine):
    result = engine.search("Riddles", {"query": "", "hitsPerPage": 3, "page": 1})
    assert result["nbHits"] == 4 and result["nbPages"] == 2 and result["page"] == 1
    assert ids(result) == ["r3"]
    assert result["hits"][0]["_rankingInfo"]["position"] == 4


def test_unretrievable_attributes_are_hidden(engine):
    hit = engine.search("Riddles", {"query": "量子"})["hits"][0]
    assert "answer" not in hit
    assert "answer" not in engine.get_object("Riddles", "r1")
    # 答えで検索してもヒットしない (検索対象外)
    assert ids(engine.search("Riddles", {"query": "シュレディンガー"})) == []


def test_attributes_to_retrieve(engine):
    hit = engine.search("Riddles", {"query": "量子", "attributesToRetrieve": ["title"]})["hits"][0]
    assert set(hit) == {"objectID", "title", "_rankingInfo"}


def test_unknown_index_returns_empty_result(engine):
    assert engine.search("Nothing", {"query": "x"})["nbHits"] == 0


def test_invalid_params_raise(engine):
    with pytest.raises(SearchError):
        engine.search("Riddles", {"query": "x", "hitsPerPage": 0})
    with pytest.raises(SearchError):
        engine.search("bad name!", {"query": "x"})


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_filters(engine):
    assert ids(engine.search("Riddles", {"query": "", "filters": "category:なぞなぞ"})) == ["r2", "r4"]
    assert ids(engine.search("Riddles", {"query": "", "filters": "tags:脱出 OR playCount>=40"})) == ["r2", "r4"]
    assert "r3" not in ids(engine.search("Riddles", {"query": "", "filters": "NOT contentType:game"}))
    assert ids(engine.search("Riddles", {"query": "", "facetFilters": [["tags:科学", "tags:脱出"]]})) == ["r4", "r1"]


def test_compile_filters_quotes_and_errors():
    pred = compile_filters('title:"a b" AND n<3')
    assert pred({"title": "a b", "n": 2}) and not pred({"title": "a b", "n": 3})
    with pytest.raises(SearchError):
        compile_filters("(a:b)")
    with pytest.raises(SearchError):
        compile_filters("nonsense")


# ---------------------------------------------------------------------------
# Updates
# ---------------------------------------------------------------------------


def test_update_and_delete_are_incremental(engine):
    engine.save_object("Riddles", {"objectID": "r1", "title": "宇宙の謎"})
    assert ids(engine.search("Riddles", {"query": "量子"})) == []
    assert ids(engine.search("Riddles", {"query": "宇宙"})) == ["r1"]
    engine.partial_update("Riddles", "r1", {"tags": ["天文"]})
    obj = engine.get_object("Riddles", "r1")
    assert obj["title"] == "宇宙の謎" and obj["tags"] == ["天文"]
    assert engine.delete_object("Riddles", "r1")
    assert ids(engine.search("Riddles", {"query": "宇宙"})) == []
    assert engine.partial_update("Riddles", "missing", {"a": 1}, create_if_missing=False) is None


def test_batch_actions(engine):
    result = engine.batch("Riddles", [
        {"action": "deleteObject", "body": {"objectID": "r2"}},
        {"action": "partialUpdateObjectNoCreate", "body": {"objectID": "zzz", "title": "x"}},
    ])
    assert result == ["r2", None]
    with pytest.raises(SearchError):
        engine.batch("Riddles", [{"action": "explode"}])
    with pytest.raises(SearchError):
        engine.save_object("Riddles", {"title": "no id"})


def test_settings_change_reindexes(engine):
    engine.set_settings("Riddles", {"searchableAttributes": ["category"]})
    assert ids(engine.search("Riddles", {"query": "量子"})) == []
    assert ids(engine.search("Riddles", {"query": "理系"})) == ["r1"]
    with pytest.raises(SearchError):
        engine.set_settings("Riddles", {"customRanking": ["up(x)"]})


# ---------------------------------------------------------------------------
# Recommend / events
# ---------------------------------------------------------------------------


def test_similar_objects(engine):
    hits = engine.recommend("Riddles", "r2", max_results=3)
    assert hits[0]["objectID"] == "r4"
    assert all(h["objectID"] != "r2" for h in hits)
    assert all("answer" not in h for h in hits)
    with pytest.raises(SearchError):
        engine.recommend("Riddles", "missing")


def test_events_boost_popular_objects(engine):
    before = ids(engine.search("Riddles", {"query": "謎解き"}))
    assert before[:2] == ["r2", "r4"]
    engine.record_events([{"eventType": "conversion", "index": "Riddles", "objectIDs": ["r4"]}] * 20)
    assert ids(engine.search("Riddles", {"query": "謎解き"}))[0] == "r4"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_snapshot_roundtrip(tmp_path):
    eng = SearchEngine(str(tmp_path))
    eng.batch("Riddles", [{"action": "addObject", "body": r} for r in RIDDLES])
    eng.record_events([{"eventType": "click", "index": "Riddles", "objectIDs": ["r3"]}])
    assert eng.flush() == 1
    assert eng.flush() == 0  # 変更がなければ書かない

    restored = SearchEngine(str(tmp_path))
    assert ids(restored.search("Riddles", {"query": "量子"})) == ["r1"]
    assert restored.get_object("Riddles", "r1")["title"] == "量子力学の謎"
    assert restored._indexes["Riddles"].popularity["r3"] == 1

    assert restored.delete_index("Riddles")
    assert not (tmp_path / "Riddles.json").exists()


def test_from_snapshot_keeps_settings():
    index = SearchIndex("Custom", {"searchableAttributes": ["body"]})
    index.save_object({"objectID": "1", "body": "量子"})
    restored = SearchIndex.from_snapshot(index.to_snapshot())
    assert restored.settings["searchableAttributes"] == ["body"]
    assert [h["objectID"] for h in restored.search({"query": "量子"})["hits"]] == ["1"]


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


@pytest.fixture
def server(engine):
    srv = create_server(engine, "127.0.0.1", 0, admin_key="admin-secret", search_key="search-key")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def call(base, method, path, body=None, key="search-key"):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("X-Qubit-Search-Key", key)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_search_and_recommend(server):
    status, body = call(server, "POST", "/1/indexes/Riddles/query", {"query": "量子", "hitsPerPage": 5})
    assert status == 200 and [h["objectID"] for h in body["hits"]] == ["r1"]
    status, body = call(server, "POST", "/1/indexes/Riddles/recommend", {"objectID": "r2"})
    assert status == 200 and body["hits"][0]["objectID"] == "r4"
    status, body = call(server, "GET", "/1/indexes/Riddles/r1")
    assert status == 200 and "answer" not in body
    status, body = call(server, "POST", "/1/events",
                        {"events": [{"eventType": "click", "index": "Riddles", "objectIDs": ["r1"]}]})
    assert status == 200 and body["counted"] == 1


def test_http_auth(server):
    assert call(server, "POST", "/1/indexes/Riddles/query", {"query": "x"}, key=None)[0] == 403
    assert call(server, "PUT", "/1/indexes/Riddles/r9", {"title": "新作"})[0] == 403
    status, body = call(server, "PUT", "/1/indexes/Riddles/r9", {"title": "新作の謎"}, key="admin-secret")
    assert status == 200 and body["objectID"] == "r9"
    status, body = call(server, "POST", "/1/indexes/Riddles/query", {"query": "新作"})
    assert [h["objectID"] for h in body["hits"]] == ["r9"]
    status, body = call(server, "DELETE", "/1/indexes/Riddles/r9", key="admin-secret")
    assert body["deleted"] is True
    assert call(server, "GET", "/health", key=None)[0] == 200


def test_http_errors(server):
    assert call(server, "POST", "/1/indexes/Riddles/query", {"query": "x", "hitsPerPage": -1})[0] == 400
    assert call(server, "GET", "/1/indexes/Riddles/missing")[0] == 404
    assert call(server, "GET", "/nope")[0] == 404
    status, body = call(server, "POST", "/1/indexes/Riddles/batch",
                        {"requests": [{"action": "addObject", "body": {"objectID": "b1", "title": "一括の謎"}}]},
                        key="admin-secret")
    assert status == 200 and body["objectIDs"] == ["b1"]
