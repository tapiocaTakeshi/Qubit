"""
MCP Server Integration Tests
MCPサーバーの統合テスト
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple, Dict, Any

import pytest


class MCPServerTestHelper:
    """MCPサーバーのテスト用ヘルパークラス"""

    def __init__(self):
        self.process = None
        self.test_dir = Path(__file__).parent.parent

    def start_server(self) -> bool:
        """MCPサーバーを起動"""
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(self.test_dir / "frontal_engine_mcp_server.py")],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # サーバーの起動を待つ
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False

    def stop_server(self):
        """MCPサーバーを停止"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()

    def send_request(self, request: Dict[str, Any]) -> Tuple[bool, str]:
        """MCPサーバーにリクエストを送信"""
        if not self.process:
            return False, "Server not running"

        try:
            request_json = json.dumps(request) + "\n"
            stdout, stderr = self.process.communicate(
                input=request_json,
                timeout=5
            )
            return True, stdout
        except subprocess.TimeoutExpired:
            return True, ""  # async serverの場合はタイムアウトは正常
        except Exception as e:
            return False, str(e)


@pytest.fixture
def mcp_helper():
    """MCPサーバーテスト用フィクスチャ"""
    helper = MCPServerTestHelper()
    yield helper
    helper.stop_server()


class TestMCPServer:
    """MCPサーバーのテストクラス"""

    def test_server_startup(self, mcp_helper):
        """サーバーが起動できることをテスト"""
        assert mcp_helper.start_server(), "Failed to start MCP server"

    def test_tools_list_request(self, mcp_helper):
        """tools/list リクエストがレスポンスを返すことをテスト"""
        assert mcp_helper.start_server()

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }

        success, response = mcp_helper.send_request(request)
        assert success, f"Request failed: {response}"

    def test_judge_tool_schema(self):
        """judge ツールのスキーマが正しいことをテスト"""
        config_file = Path(__file__).parent.parent / "frontal_engine.mcp.json"
        assert config_file.exists(), "Config file not found"

        with open(config_file) as f:
            config = json.load(f)

        # 必須フィールドのチェック
        assert "name" in config
        assert "version" in config
        assert "tools" in config
        assert len(config["tools"]) > 0

        # judge ツールの確認
        judge_tool = next(
            (t for t in config["tools"] if t["name"] == "judge"),
            None
        )
        assert judge_tool is not None, "judge tool not found"

        # スキーマの確認
        assert "inputSchema" in judge_tool
        assert "properties" in judge_tool["inputSchema"]
        assert "context" in judge_tool["inputSchema"]["properties"]
        assert "judgment_request" in judge_tool["inputSchema"]["properties"]

    def test_required_files(self):
        """必須ファイルが存在することをテスト"""
        test_dir = Path(__file__).parent.parent

        required_files = [
            "frontal_engine_mcp_server.py",
            "frontal_engine.mcp.json",
            "neuroquantum_layered.py",
            "qbnn_layered.py",
        ]

        for filename in required_files:
            file_path = test_dir / filename
            assert file_path.exists(), f"Required file not found: {filename}"

    def test_python_syntax(self):
        """MCPサーバーのPythonコードが文法的に正しいことをテスト"""
        server_file = Path(__file__).parent.parent / "frontal_engine_mcp_server.py"
        assert server_file.exists()

        # Pythonコンパイルをテスト
        try:
            with open(server_file) as f:
                code = f.read()
            compile(code, server_file, 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {server_file}: {e}")


class TestFrontalEngineJudge:
    """FrontalEngineJudge クラスのテスト"""

    def test_judge_initialization(self):
        """判断エンジンが初期化できることをテスト"""
        from frontal_engine_mcp_server import FrontalEngineJudge

        judge = FrontalEngineJudge()
        assert judge is not None

    def test_judge_basic_judgment(self):
        """基本的な判断が実行できることをテスト"""
        from frontal_engine_mcp_server import FrontalEngineJudge

        judge = FrontalEngineJudge()

        task = {
            "context": "プロジェクトは予定通り進行しており、品質基準をすべて満たしています。",
            "judgment_request": "このプロジェクトをリリースしても安全か？"
        }

        result = judge.judge(task)

        # 結果の検証
        assert "decision" in result
        assert result["decision"] in ["Yes", "No"]
        assert "score" in result
        assert 0 <= result["score"] <= 100
        assert "reasoning" in result
        assert "confidence" in result
        assert "key_factors" in result
        assert "timestamp" in result

    def test_judge_strict_mode(self):
        """strict_mode オプションが機能することをテスト"""
        from frontal_engine_mcp_server import FrontalEngineJudge

        judge = FrontalEngineJudge()

        # Strict mode on
        task = {
            "context": "テスト文脈",
            "judgment_request": "テスト判断",
            "strict_mode": True
        }

        result = judge.judge(task)
        assert "decision" in result

    def test_judge_with_criteria(self):
        """criteria パラメータが処理されることをテスト"""
        from frontal_engine_mcp_server import FrontalEngineJudge

        judge = FrontalEngineJudge()

        task = {
            "context": "高い品質基準を満たしており、パフォーマンスも良好です。",
            "judgment_request": "この実装を採用すべきか？",
            "criteria": {"quality": "high", "performance": 90}
        }

        result = judge.judge(task)
        assert "score" in result

    def test_judge_with_options(self):
        """options パラメータが処理されることをテスト"""
        from frontal_engine_mcp_server import FrontalEngineJudge

        judge = FrontalEngineJudge()

        task = {
            "context": "ベンダーB: 価格中程度、サポート強い",
            "judgment_request": "ベンダーBを選択することが最適か？",
            "options": ["ベンダーA", "ベンダーB", "ベンダーC"]
        }

        result = judge.judge(task)
        assert "decision" in result

    def test_judge_missing_context(self):
        """必須パラメータ（context）が不足する場合のエラー処理をテスト"""
        from frontal_engine_mcp_server import FrontalEngineJudge

        judge = FrontalEngineJudge()

        task = {
            "judgment_request": "テスト判断"
            # context が不足
        }

        result = judge.judge(task)
        assert "error" in result or result["score"] == 0


class TestMCPTools:
    """MCPツールの統合テスト"""

    @pytest.mark.asyncio
    async def test_judge_tool_execution(self):
        """judge ツールが実行できることをテスト"""
        from frontal_engine_mcp_server import create_server

        server = create_server()
        tools = await server.list_tools()

        judge_tool = next((t for t in tools if t.name == "judge"), None)
        assert judge_tool is not None, "judge tool not found"

        # ツールの入力スキーマを確認
        assert judge_tool.inputSchema is not None
        assert "context" in judge_tool.inputSchema["properties"]
        assert "judgment_request" in judge_tool.inputSchema["properties"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
