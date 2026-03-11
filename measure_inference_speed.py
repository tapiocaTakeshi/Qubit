"""
neuroQ API 推論速度測定スクリプト
HuggingFace Inference Endpoint の推論レイテンシとスループットを計測する
"""

import requests
import time
import json
import statistics

API_URL = "https://buy78wjaoq4vyn8f.us-east4.gcp.endpoints.huggingface.cloud"

# テストプロンプト（短・中・長）
TEST_PROMPTS = [
    {"label": "短文 (short)", "text": "量子"},
    {"label": "中文 (medium)", "text": "量子コンピュータの未来について"},
    {"label": "長文 (long)", "text": "人工知能と量子力学の融合は、次世代の計算技術において重要な役割を果たすと考えられています。"},
]

# トークン数パターン
TOKEN_COUNTS = [50, 100, 200]


def measure_single_request(prompt: str, max_new_tokens: int = 100, temperature: float = 0.8) -> dict:
    """単一リクエストの推論速度を測定"""
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        },
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        end_time = time.perf_counter()

        latency = end_time - start_time

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", "")
            else:
                generated_text = str(result)

            return {
                "success": True,
                "latency_sec": latency,
                "generated_text": generated_text,
                "output_length": len(generated_text),
                "status_code": response.status_code,
            }
        else:
            return {
                "success": False,
                "latency_sec": latency,
                "status_code": response.status_code,
                "error": response.text[:300],
            }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout (120s)", "latency_sec": 120.0}
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "error": f"Connection error: {e}", "latency_sec": 0.0}


def run_benchmark():
    """推論速度ベンチマークを実行"""
    print("=" * 70)
    print("  neuroQ API 推論速度測定 (Inference Speed Benchmark)")
    print(f"  Endpoint: {API_URL}")
    print("=" * 70)

    # 1. ウォームアップリクエスト
    print("\n[1/3] ウォームアップ (Warm-up request)...")
    warmup = measure_single_request("テスト", max_new_tokens=10)
    if not warmup["success"]:
        print(f"  ウォームアップ失敗: {warmup.get('error', warmup.get('status_code'))}")
        print("  APIが利用可能か確認してください。")
        return
    print(f"  ウォームアップ完了: {warmup['latency_sec']:.3f}s")

    # 2. プロンプト別レイテンシ測定
    print("\n[2/3] プロンプト別レイテンシ測定 (max_new_tokens=100, 各3回)")
    print("-" * 70)
    print(f"{'プロンプト':<20} {'平均(s)':<10} {'最小(s)':<10} {'最大(s)':<10} {'標準偏差':<10}")
    print("-" * 70)

    all_results = []
    for prompt_info in TEST_PROMPTS:
        latencies = []
        for i in range(3):
            result = measure_single_request(prompt_info["text"], max_new_tokens=100)
            if result["success"]:
                latencies.append(result["latency_sec"])
                if i == 0:
                    all_results.append({
                        "prompt": prompt_info,
                        "sample_output": result["generated_text"][:100],
                    })

        if latencies:
            avg = statistics.mean(latencies)
            std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            print(f"{prompt_info['label']:<20} {avg:<10.3f} {min(latencies):<10.3f} {max(latencies):<10.3f} {std:<10.3f}")
        else:
            print(f"{prompt_info['label']:<20} {'FAILED':<10}")

    # 3. トークン数別スループット測定
    print(f"\n[3/3] トークン数別スループット測定 (prompt='量子コンピュータ', 各2回)")
    print("-" * 70)
    print(f"{'max_new_tokens':<18} {'平均レイテンシ(s)':<18} {'推定tokens/s':<15}")
    print("-" * 70)

    for token_count in TOKEN_COUNTS:
        latencies = []
        for _ in range(2):
            result = measure_single_request("量子コンピュータ", max_new_tokens=token_count)
            if result["success"]:
                latencies.append(result["latency_sec"])

        if latencies:
            avg = statistics.mean(latencies)
            tokens_per_sec = token_count / avg
            print(f"{token_count:<18} {avg:<18.3f} {tokens_per_sec:<15.1f}")
        else:
            print(f"{token_count:<18} {'FAILED':<18}")

    # サンプル出力
    print("\n" + "=" * 70)
    print("  サンプル出力 (Sample Outputs)")
    print("=" * 70)
    for r in all_results:
        print(f"\n  [{r['prompt']['label']}] \"{r['prompt']['text']}\"")
        print(f"  → {r['sample_output']}...")

    print("\n" + "=" * 70)
    print("  測定完了 (Benchmark Complete)")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
