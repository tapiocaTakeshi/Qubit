#!/usr/bin/env python3
"""
QBNN Frontal Engine - REST API Server
FastAPI ベースの REST API で Judge機能を提供
"""

import json
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    BaseModel = object
    Field = None
    uvicorn = None

from frontal_engine_mcp_server import FrontalEngineJudge


# ============================================================
# Request/Response Models
# ============================================================

class JudgeRequest(BaseModel):
    """判断リクエストのスキーマ"""
    context: str = Field(..., description="判断の背景情報・文脈（必須）")
    judgment_request: str = Field(..., description="何を判断するか・判断内容（必須）")
    criteria: Optional[Dict[str, Any]] = Field(
        default=None,
        description="判断基準（オプション）"
    )
    options: Optional[list] = Field(
        default=None,
        description="検討対象となる選択肢（オプション）"
    )
    strict_mode: bool = Field(
        default=False,
        description="厳密な判断モード"
    )


class JudgeResponse(BaseModel):
    """判断レスポンスのスキーマ"""
    decision: str = Field(..., description="Yes/No判断")
    score: int = Field(..., description="判断スコア（0-100）")
    reasoning: str = Field(..., description="判断の根拠説明")
    confidence: str = Field(..., description="判断の確信度（high/medium/low）")
    key_factors: list = Field(..., description="判断に影響を与えた主要要因")
    timestamp: str = Field(..., description="ISO 8601形式のタイムスタンプ")


# ============================================================
# FastAPI Server
# ============================================================

def create_app():
    """FastAPI アプリケーションを作成"""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Install: pip install fastapi uvicorn")

    app = FastAPI(
        title="QBNN Frontal Engine API",
        description="脳の前頭葉の判断機能を提供するREST API",
        version="1.0.0"
    )

    # グローバルインスタンス
    judge_engine = FrontalEngineJudge()

    # ============================================================
    # Endpoints
    # ============================================================

    @app.get("/")
    async def root():
        """ルートエンドポイント"""
        return {
            "service": "QBNN Frontal Engine API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "judge": "/judge",
                "docs": "/docs"
            }
        }

    @app.get("/health")
    async def health():
        """ヘルスチェック"""
        return {
            "status": "healthy",
            "service": "QBNN Frontal Engine",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    @app.post("/judge", response_model=JudgeResponse)
    async def judge_endpoint(request: JudgeRequest):
        """
        判断エンドポイント
        前頭葉の判断機能を使用してYes/No決定を返す
        """
        try:
            # リクエストを辞書に変換
            task = {
                "context": request.context,
                "judgment_request": request.judgment_request,
                "criteria": request.criteria or {},
                "options": request.options or [],
                "strict_mode": request.strict_mode
            }

            # 判断を実行
            result = judge_engine.judge(task)

            # エラーチェック
            if result.get("error"):
                raise HTTPException(
                    status_code=400,
                    detail=result.get("reasoning", "Unknown error")
                )

            return JudgeResponse(**result)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Judge execution error: {str(e)}"
            )

    @app.post("/judge/batch")
    async def judge_batch(requests: list[JudgeRequest]):
        """
        バッチ判断エンドポイント
        複数の判断タスクを一度に処理
        """
        results = []
        for req in requests:
            task = {
                "context": req.context,
                "judgment_request": req.judgment_request,
                "criteria": req.criteria or {},
                "options": req.options or [],
                "strict_mode": req.strict_mode
            }
            result = judge_engine.judge(task)
            results.append(result)

        return {
            "count": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    return app


# ============================================================
# Main
# ============================================================

def main():
    """REST API サーバーを起動"""
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI is required")
        print("Install: pip install fastapi uvicorn")
        return

    app = create_app()
    print("🚀 Starting QBNN Frontal Engine REST API")
    print("📍 Server: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
