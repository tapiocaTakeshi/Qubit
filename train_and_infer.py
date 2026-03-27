#!/usr/bin/env python3
"""
学習 → 推論 一括実行スクリプト

HuggingFace Inference Endpoint に対して:
  1. QAデータで学習 (train_qa)
  2. 推論テスト (inference)
を順番に実行する。

使い方:
  # デフォルト: QA学習 → 推論テスト
  python train_and_infer.py

  # 推論のみ
  python train_and_infer.py --skip-train --prompt "量子コンピュータとは"

  # 学習のみ (推論スキップ)
  python train_and_infer.py --skip-infer

  # カスタムプロンプト複数指定
  python train_and_infer.py --prompt "日本の首都は" --prompt "人工知能とは"

  # エポック数・学習率を調整
  python train_and_infer.py --epochs 8 --lr 5e-5
"""
import argparse
import json
import os
import sys
import time
import requests

ENDPOINT_URL = "https://vvcci2ps4y3wfx7m.us-east4.gcp.endpoints.huggingface.cloud"

# ── QA学習データ ──────────────────────────────────────────────
QA_PAIRS = [
    {"question": "日本の首都はどこですか？", "answer": "日本の首都は東京です。東京は政治、経済、文化の中心地であり、世界最大級の都市圏を形成しています。"},
    {"question": "富士山の高さは何メートルですか？", "answer": "富士山の高さは3776メートルです。日本で最も高い山であり、世界遺産に登録されています。"},
    {"question": "太陽系で一番大きい惑星は何ですか？", "answer": "木星が太陽系で最も大きい惑星です。直径は約14万キロメートルで、地球の約11倍の大きさがあります。"},
    {"question": "水の化学式は何ですか？", "answer": "水の化学式はH2Oです。水素原子2つと酸素原子1つから構成されています。"},
    {"question": "光の速さはどれくらいですか？", "answer": "光の速さは秒速約30万キロメートル（299,792,458 m/s）です。宇宙で最も速い速度です。"},
    {"question": "日本で一番長い川は何ですか？", "answer": "信濃川が日本で最も長い川で、全長367キロメートルです。新潟県と長野県を流れています。"},
    {"question": "地球の年齢はどれくらいですか？", "answer": "地球の年齢は約46億年です。太陽系の形成とほぼ同時期に誕生しました。"},
    {"question": "人間の体で一番大きい臓器は何ですか？", "answer": "皮膚が人間の体で最も大きい臓器です。体全体を覆い、外部からの保護や体温調節を行います。"},
    {"question": "DNAとは何ですか？", "answer": "DNAはデオキシリボ核酸の略で、遺伝情報を保持する分子です。二重らせん構造を持ち、すべての生物の設計図となっています。"},
    {"question": "重力とは何ですか？", "answer": "重力とは、質量を持つ物体同士が引き合う力です。ニュートンの万有引力の法則で説明され、地球上では約9.8m/s²の加速度を生じます。"},
    {"question": "光合成とは何ですか？", "answer": "光合成とは、植物が光エネルギーを使って二酸化炭素と水からブドウ糖と酸素を作り出す反応です。"},
    {"question": "量子コンピュータとは何ですか？", "answer": "量子コンピュータは量子力学の原理（重ね合わせやエンタングルメント）を利用して計算を行うコンピュータです。従来のコンピュータでは困難な問題を高速に解ける可能性があります。"},
    {"question": "人工知能とは何ですか？", "answer": "人工知能（AI）は、人間の知的能力をコンピュータで実現する技術です。機械学習やディープラーニングなどの手法があります。"},
    {"question": "プログラミングとは何ですか？", "answer": "プログラミングとは、コンピュータに実行させたい処理をプログラミング言語で記述することです。Python、Java、C++などの言語があります。"},
    {"question": "俳句とは何ですか？", "answer": "俳句は五七五の17音で詠む日本の伝統的な短詩です。季語を含むことが基本的なルールです。松尾芭蕉が有名です。"},
    {"question": "明治維新は何年に起きましたか？", "answer": "明治維新は1868年に起きました。江戸幕府が倒れ、天皇中心の近代国家が成立しました。"},
    {"question": "円周率とは何ですか？", "answer": "円周率は円の周の長さと直径の比で、約3.14159です。πという記号で表され、無理数です。"},
    {"question": "素数とは何ですか？", "answer": "素数とは、1とその数自身でしか割り切れない2以上の自然数です。2、3、5、7、11などが素数です。"},
    {"question": "ピタゴラスの定理とは何ですか？", "answer": "ピタゴラスの定理は、直角三角形において斜辺の2乗が他の2辺の2乗の和に等しいという定理です。a²+b²=c²で表されます。"},
    {"question": "健康のために大切なことは何ですか？", "answer": "健康のためには、バランスの良い食事、適度な運動、十分な睡眠が大切です。また、ストレス管理と定期的な健康診断も重要です。"},
]

# ── デフォルト推論プロンプト ────────────────────────────────────
DEFAULT_PROMPTS = [
    "日本の首都は",
    "量子コンピュータとは",
    "人工知能の未来は",
    "プログラミングを学ぶには",
    "富士山について",
]


# ── HTTP helper ───────────────────────────────────────────────

def _post(endpoint_url, payload, timeout=600, retries=3):
    """POST request with retries and exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                endpoint_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list):
                    result = result[0]
                return result
            elif resp.status_code == 503:
                wait = 10 * (attempt + 1)
                print(f"    モデルロード中、{wait}秒待機...")
                time.sleep(wait)
                continue
            else:
                print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"    タイムアウト (試行 {attempt+1}/{retries})")
        except Exception as e:
            print(f"    エラー: {e}")
        if attempt < retries - 1:
            wait = 2 ** (attempt + 1)
            print(f"    {wait}秒後にリトライ...")
            time.sleep(wait)

    return {"error": "全てのリトライに失敗しました"}


# ── ステータス確認 ────────────────────────────────────────────

def check_status(endpoint_url):
    """エンドポイントのステータスを確認"""
    print("\n[Status] エンドポイント状態を確認中...")
    result = _post(endpoint_url, {
        "inputs": "__status__",
        "parameters": {"action": "status"}
    }, timeout=30)
    return result


# ── 学習 (train_qa) ──────────────────────────────────────────

def train_qa(endpoint_url, qa_pairs, epochs=4, lr=3e-5, batch_size=4,
             grad_accum_steps=4, repeat=3):
    """QAデータで学習を実行"""
    print(f"\n{'='*60}")
    print(f"  [Train] QA学習開始")
    print(f"  QAペア数: {len(qa_pairs)}")
    print(f"  エポック: {epochs}, 学習率: {lr}, リピート: {repeat}")
    print(f"{'='*60}")

    chunk_size = max(len(qa_pairs) // 2, 1)
    chunks = [qa_pairs[i:i+chunk_size] for i in range(0, len(qa_pairs), chunk_size)]

    results = []
    for i, chunk in enumerate(chunks):
        print(f"\n  チャンク {i+1}/{len(chunks)} ({len(chunk)} QAペア)...")
        result = _post(endpoint_url, {
            "inputs": "__train_qa__",
            "parameters": {
                "action": "train_qa",
                "qa_pairs": chunk,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "repeat": repeat,
            }
        })
        status = result.get("status", result.get("error", "unknown"))
        final_loss = result.get("final_loss", "N/A")
        print(f"    状態: {status}, 最終Loss: {final_loss}")
        results.append(result)

    return results


# ── 推論 (inference) ─────────────────────────────────────────

def run_inference(endpoint_url, prompt, max_new_tokens=100, temperature=0.7,
                  top_k=40, top_p=0.9, repetition_penalty=1.3):
    """テキスト生成推論を実行"""
    result = _post(endpoint_url, {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
    }, timeout=60)
    return result


# ── メイン ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="学習→推論 一括実行スクリプト (HF Inference Endpoint)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--endpoint", default=ENDPOINT_URL, help="Endpoint URL")
    parser.add_argument("--skip-train", action="store_true", help="学習をスキップ")
    parser.add_argument("--skip-infer", action="store_true", help="推論をスキップ")
    parser.add_argument("--prompt", action="append", help="推論プロンプト (複数指定可)")
    parser.add_argument("--epochs", type=int, default=4, help="学習エポック数")
    parser.add_argument("--lr", type=float, default=3e-5, help="学習率")
    parser.add_argument("--repeat", type=int, default=3, help="QAデータリピート回数")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="最大生成トークン数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS

    print("=" * 60)
    print("  neuroQ: 学習→推論 一括実行")
    print(f"  エンドポイント: {args.endpoint}")
    print(f"  学習: {'スキップ' if args.skip_train else f'QA ({len(QA_PAIRS)}ペア, {args.epochs}エポック)'}")
    print(f"  推論: {'スキップ' if args.skip_infer else f'{len(prompts)}プロンプト'}")
    print("=" * 60)

    # ── Step 0: ステータス確認 ─────────────────────────────────
    status = check_status(args.endpoint)
    if "error" in status:
        print(f"\n  エンドポイントに接続できません: {status['error']}")
        print("  エンドポイントが起動しているか確認してください。")
        sys.exit(1)
    print(f"  モデル: {status.get('model', 'unknown')}")
    print(f"  パラメータ数: {status.get('parameters', 'unknown')}")

    # ── Step 1: 学習 ──────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("  STEP 1: QA学習")
        print("=" * 60)
        train_results = train_qa(
            args.endpoint,
            QA_PAIRS,
            epochs=args.epochs,
            lr=args.lr,
            repeat=args.repeat,
        )
        success = sum(1 for r in train_results if r.get("status") == "training_complete")
        print(f"\n  学習完了: {success}/{len(train_results)} チャンク成功")
    else:
        print("\n  [Skip] 学習をスキップしました")

    # ── Step 2: 推論 ──────────────────────────────────────────
    if not args.skip_infer:
        print("\n" + "=" * 60)
        print("  STEP 2: 推論テスト")
        print("=" * 60)

        for i, prompt in enumerate(prompts):
            print(f"\n  [{i+1}/{len(prompts)}] プロンプト: {prompt}")
            result = run_inference(
                args.endpoint,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            generated = result.get("generated_text", result.get("error", str(result)))
            print(f"  生成結果: {generated}")
    else:
        print("\n  [Skip] 推論をスキップしました")

    # ── 完了 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
