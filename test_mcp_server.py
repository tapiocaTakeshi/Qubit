#!/usr/bin/env python3
"""
QBNN Frontal Engine MCP Server - クライアント動作確認
FastMCPサーバーのツール定義が正しく設定されているか確認
"""

import json
from frontal_engine_mcp_server import judge_tool


def test_mcp_judge_tool():
    """MCPの judge_tool が正しく動作するかテスト"""
    print("=" * 60)
    print("MCP judge_tool テスト")
    print("=" * 60)

    # テスト1: 基本的な判断
    result = judge_tool(
        context="プロジェクトは予定通り進行しており、品質基準をすべて満たしています。",
        judgment_request="このプロジェクトをリリースしても安全か？"
    )
    print("\n【テスト1】基本判断:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert "decision" in result
    assert "score" in result
    assert "reasoning" in result
    assert "confidence" in result
    assert "key_factors" in result
    assert "timestamp" in result

    # テスト2: 基準付き判断
    result = judge_tool(
        context="提案者の信頼スコア: 85/100、過去の成功率: 80%",
        judgment_request="この提案を承認すべきか？",
        criteria={"trust_score": 80, "success_rate": "80%"}
    )
    print("\n【テスト2】基準付き判断:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert result["score"] > 50

    # テスト3: 厳密モード
    result = judge_tool(
        context="新技術導入にはいくつかのリスクがあります",
        judgment_request="リスクは許容可能か？",
        strict_mode=True
    )
    print("\n【テスト3】厳密モード判断:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert isinstance(result["decision"], str)
    assert result["decision"] in ["Yes", "No"]

    # テスト4: 複数選択肢
    result = judge_tool(
        context="ベンダーA: 安い。ベンダーB: 中程度、サポート強い。ベンダーC: 高い",
        judgment_request="ベンダーBが最適か？",
        options=["ベンダーA", "ベンダーB", "ベンダーC"]
    )
    print("\n【テスト4】複数選択肢:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("✓ すべてのMCPテストが成功しました")
    print("=" * 60)


if __name__ == "__main__":
    test_mcp_judge_tool()
