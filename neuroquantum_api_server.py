#!/usr/bin/env python3
"""
NeuroQuantum REST API Server

neuroquantum_layered.py の推論エンジンを REST API として公開し、
TypeScript/Node.js から量子インスパイアド推論にアクセスできるようにします。
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

# neuroquantum_layered をインポート
try:
    from neuroquantum_layered import (
        NeuroQuantumConfig,
        get_gpu_adaptive_config,
        get_model_config_by_size,
    )
    NEUROQUANTUM_AVAILABLE = True
except ImportError:
    NEUROQUANTUM_AVAILABLE = False
    print("⚠️  neuroquantum_layered が見つかりません")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask アプリケーション
app = Flask(__name__)
CORS(app)

# API バージョン
API_VERSION = "1.0.0"

@dataclass
class JudgmentRequest:
    """推論リクエスト"""
    action: str
    context: str
    judgment_type: str = "safety"
    strict_mode: bool = False

@dataclass
class JudgmentResponse:
    """推論レスポンス"""
    decision: str  # "Yes" or "No"
    score: float  # 0-100
    reasoning: str
    confidence: str  # "high", "medium", "low"
    factors: List[str]
    timestamp: str
    system: str = "neuroquantum"
    processing_time_ms: float = 0.0

class NeuroQuantumInferenceEngine:
    """neuroquantum_layered.py ベースの推論エンジン"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        推論エンジンを初期化

        Args:
            config: モデル設定 (None の場合は GPU 適応設定を使用)
        """
        self.config = config or get_gpu_adaptive_config()
        logger.info(f"NeuroQuantum エンジン初期化: {self.config.get('gpu_tier', 'unknown')} tier")

        # TODO: 実際の neuroquantum モデルをロード
        # self.model = NeuroQuantumModel(self.config)

    def judge(
        self,
        action: str,
        context: str,
        judgment_type: str = "safety",
        strict_mode: bool = False
    ) -> JudgmentResponse:
        """
        量子インスパイアド推論で判定を実行

        Args:
            action: 評価対象の行動
            context: 状況文脈
            judgment_type: 判定タイプ (safety, ethics, quality, risk, decision, priority)
            strict_mode: 厳格モード (スコア >= 70 で Yes)

        Returns:
            JudgmentResponse: 推論結果
        """
        import time
        from datetime import datetime

        start_time = time.time()

        # TODO: 実際の推論ロジック
        # 現在はダミー実装

        # 簡易的なキーワード分析で仮のスコアを生成
        positive_keywords = ["safe", "good", "ethical", "positive", "yes"]
        negative_keywords = ["unsafe", "bad", "unethical", "negative", "no"]

        text = (action + " " + context).lower()
        positive_count = sum(1 for kw in positive_keywords if kw in text)
        negative_count = sum(1 for kw in negative_keywords if kw in text)

        # スコア計算
        if positive_count + negative_count == 0:
            score = 50.0
        else:
            score = (positive_count / (positive_count + negative_count)) * 100

        # 厳格モード適用
        if strict_mode and score >= 50 and score < 70:
            decision = "No"
        elif strict_mode and score < 50:
            decision = "No"
        else:
            decision = "Yes" if score >= 50 else "No"

        # 信頼度判定
        if score >= 80 or score <= 20:
            confidence = "high"
        elif score >= 65 or score <= 35:
            confidence = "medium"
        else:
            confidence = "low"

        # 推論時間
        processing_time_ms = (time.time() - start_time) * 1000

        return JudgmentResponse(
            decision=decision,
            score=round(score, 2),
            reasoning=f"{judgment_type} judgment: {context}",
            confidence=confidence,
            factors=[judgment_type, "neural_analysis", "quantum_inspired"],
            timestamp=datetime.utcnow().isoformat() + "Z",
            system="neuroquantum",
            processing_time_ms=round(processing_time_ms, 2)
        )

# グローバルエンジンインスタンス
inference_engine = None

def get_inference_engine() -> NeuroQuantumInferenceEngine:
    """推論エンジンを取得（シングルトン）"""
    global inference_engine
    if inference_engine is None:
        inference_engine = NeuroQuantumInferenceEngine()
    return inference_engine

# ========================================
# REST API エンドポイント
# ========================================

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """ヘルスチェック"""
    return jsonify({
        "status": "healthy",
        "version": API_VERSION,
        "neuroquantum_available": NEUROQUANTUM_AVAILABLE,
        "gpu_info": get_inference_engine().config.get("gpu_info", {})
    }), 200

@app.route('/api/v1/config', methods=['GET'])
def get_config():
    """現在の設定を返す"""
    config = get_inference_engine().config.copy()
    # GPU情報を削除（深刻なシリアライゼーション問題を避けるため）
    config.pop('gpu_info', None)

    return jsonify({
        "config": config
    }), 200

@app.route('/api/v1/judge', methods=['POST'])
def judge_endpoint():
    """判定エンドポイント"""
    try:
        data = request.get_json()

        # リクエスト検証
        if not data or 'action' not in data or 'context' not in data:
            return jsonify({
                "error": "Missing required fields: 'action' and 'context'"
            }), 400

        # パラメータ抽出
        action = data.get('action', '')
        context = data.get('context', '')
        judgment_type = data.get('judgment_type', 'safety')
        strict_mode = data.get('strict_mode', False)

        # 推論実行
        engine = get_inference_engine()
        result = engine.judge(action, context, judgment_type, strict_mode)

        return jsonify(asdict(result)), 200

    except Exception as e:
        logger.error(f"判定エラー: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/v1/batch_judge', methods=['POST'])
def batch_judge_endpoint():
    """バッチ判定エンドポイント"""
    try:
        data = request.get_json()

        if not data or 'requests' not in data:
            return jsonify({
                "error": "Missing required field: 'requests'"
            }), 400

        requests_list = data.get('requests', [])
        if not isinstance(requests_list, list):
            return jsonify({
                "error": "'requests' must be a list"
            }), 400

        # バッチ処理
        engine = get_inference_engine()
        results = []

        for req in requests_list:
            result = engine.judge(
                action=req.get('action', ''),
                context=req.get('context', ''),
                judgment_type=req.get('judgment_type', 'safety'),
                strict_mode=req.get('strict_mode', False)
            )
            results.append(asdict(result))

        return jsonify({
            "results": results,
            "count": len(results)
        }), 200

    except Exception as e:
        logger.error(f"バッチ判定エラー: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/v1/safety_check', methods=['POST'])
def safety_check_endpoint():
    """安全性チェックエンドポイント"""
    try:
        data = request.get_json()

        if not data or 'action' not in data:
            return jsonify({
                "error": "Missing required field: 'action'"
            }), 400

        action = data.get('action', '')
        context = data.get('context', '')
        risks = data.get('risks', [])

        # リスク情報をコンテキストに追加
        if risks:
            context += f"\n\nRisks: {', '.join(risks)}"

        engine = get_inference_engine()
        result = engine.judge(action, context, 'safety', False)

        return jsonify({
            "safe": result.decision == "Yes",
            "result": asdict(result)
        }), 200

    except Exception as e:
        logger.error(f"安全性チェックエラー: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/v1/ethics_check', methods=['POST'])
def ethics_check_endpoint():
    """倫理チェックエンドポイント"""
    try:
        data = request.get_json()

        if not data or 'action' not in data:
            return jsonify({
                "error": "Missing required field: 'action'"
            }), 400

        action = data.get('action', '')
        stakeholders = data.get('stakeholders', [])
        harms = data.get('potential_harms', [])

        # コンテキスト構築
        context = "倫理的評価"
        if stakeholders:
            context += f"\n\nStakeholders: {', '.join(stakeholders)}"
        if harms:
            context += f"\n\nPotential harms: {', '.join(harms)}"

        engine = get_inference_engine()
        result = engine.judge(action, context, 'ethics', False)

        return jsonify(asdict(result)), 200

    except Exception as e:
        logger.error(f"倫理チェックエラー: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/v1/quality_eval', methods=['POST'])
def quality_eval_endpoint():
    """品質評価エンドポイント"""
    try:
        data = request.get_json()

        if not data or 'content' not in data:
            return jsonify({
                "error": "Missing required field: 'content'"
            }), 400

        content = data.get('content', '')
        requirements = data.get('requirements', [])
        intent = data.get('user_intent', '')

        # コンテキスト構築
        context = "品質評価"
        if requirements:
            context += f"\n\nRequirements: {', '.join(requirements)}"
        if intent:
            context += f"\n\nIntent: {intent}"

        engine = get_inference_engine()
        result = engine.judge(content, context, 'quality', False)

        return jsonify(asdict(result)), 200

    except Exception as e:
        logger.error(f"品質評価エラー: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/v1/status', methods=['GET'])
def status_endpoint():
    """サーバーステータス"""
    engine = get_inference_engine()
    config = engine.config.copy()
    config.pop('gpu_info', None)

    return jsonify({
        "status": "running",
        "version": API_VERSION,
        "model_config": config,
        "neuroquantum_available": NEUROQUANTUM_AVAILABLE
    }), 200

@app.errorhandler(404)
def not_found(error):
    """404 エラー"""
    return jsonify({
        "error": "Not found",
        "path": request.path
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """500 エラー"""
    return jsonify({
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NeuroQuantum REST API Server')
    parser.add_argument('--host', default='127.0.0.1', help='Bind address')
    parser.add_argument('--port', type=int, default=5000, help='Port')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--gpu-tier', choices=['ultra', 'high', 'mid', 'low', 'cpu'],
                        help='Force GPU tier')

    args = parser.parse_args()

    # GPU ティアを強制する場合
    if args.gpu_tier:
        logger.info(f"GPU ティアを強制: {args.gpu_tier}")

    logger.info(f"NeuroQuantum REST API Server を起動します")
    logger.info(f"  アドレス: {args.host}:{args.port}")
    logger.info(f"  Debug: {args.debug}")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )
