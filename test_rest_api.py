#!/usr/bin/env python3
"""
QBNN Frontal Engine REST API - テストスクリプト
FastAPI サーバーをテストモードで実行
"""

import json
from frontal_engine_api import create_app
from fastapi.testclient import TestClient


def test_rest_api():
    """REST API をテストモードで実行"""
    print("=" * 60)
    print("QBNN Frontal Engine REST API テスト")
    print("=" * 60)

    # テストクライアント作成
    app = create_app()
    client = TestClient(app)

    # テスト1: ルートエンドポイント
    print("\n【テスト1】ルートエンドポイント")
    response = client.get("/")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    assert response.status_code == 200

    # テスト2: ヘルスチェック
    print("\n【テスト2】ヘルスチェック")
    response = client.get("/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    # テスト3: 基本判断
    print("\n【テスト3】基本判断")
    payload = {
        "context": "プロジェクトは予定通り進行しており、品質基準をすべて満たしています。",
        "judgment_request": "このプロジェクトをリリースしても安全か？"
    }
    response = client.post("/judge", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert response.status_code == 200
    assert "decision" in result
    assert "score" in result
    assert "reasoning" in result

    # テスト4: 基準付き判断
    print("\n【テスト4】基準付き判断")
    payload = {
        "context": "提案者の信頼スコア: 85/100、過去の成功率: 80%",
        "judgment_request": "この提案を承認すべきか？",
        "criteria": {"trust_score": 80, "success_rate": "80%"}
    }
    response = client.post("/judge", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert response.status_code == 200
    assert result["decision"] in ["Yes", "No"]

    # テスト5: 厳密モード
    print("\n【テスト5】厳密モード")
    payload = {
        "context": "新技術導入にはいくつかのリスクがあります",
        "judgment_request": "リスクは許容可能か？",
        "strict_mode": True
    }
    response = client.post("/judge", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert response.status_code == 200

    # テスト6: 複数選択肢
    print("\n【テスト6】複数選択肢")
    payload = {
        "context": "ベンダーA: 安い。ベンダーB: 中程度、サポート強い。ベンダーC: 高い",
        "judgment_request": "ベンダーBが最適か？",
        "options": ["ベンダーA", "ベンダーB", "ベンダーC"]
    }
    response = client.post("/judge", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert response.status_code == 200

    # テスト7: バッチ判断
    print("\n【テスト7】バッチ判断")
    payload = [
        {
            "context": "プロジェクト1: 品質95%",
            "judgment_request": "リリース可能か？"
        },
        {
            "context": "プロジェクト2: 品質80%",
            "judgment_request": "リリース可能か？"
        }
    ]
    response = client.post("/judge/batch", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert response.status_code == 200
    assert result["count"] == 2

    # テスト8: エラーハンドリング
    print("\n【テスト8】エラーハンドリング（必須パラメータなし）")
    payload = {"context": "これは不完全な入力です"}
    response = client.post("/judge", json=payload)
    print(f"Status: {response.status_code}")
    assert response.status_code == 422  # Validation Error

    print("\n" + "=" * 60)
    print("✓ すべてのREST APIテストが成功しました")
    print("=" * 60)


if __name__ == "__main__":
    test_rest_api()
