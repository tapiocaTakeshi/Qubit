#!/usr/bin/env python3
"""
リモート学習クライアント - HF Inference Endpointのtrain APIを叩いて学習させる

アクション一覧:
  inference  (デフォルト) — テキスト生成
  train                  — 一般データセットによる学習
  train_qa               — QA形式データでの学習
  train_texts            — テキストデータを直接送信して学習
  status                 — 学習状況の確認

使い方:
  # ステータス確認
  python train_remote.py --mode status

  # QAデータでリモート学習 (train_qa API)
  python train_remote.py --mode qa

  # データセットでリモート学習 (train API)
  python train_remote.py --mode dataset --dataset_id fujiki/japanese_alpaca_data

  # ローカルのテキストファイルを送信して学習 (train_texts API)
  python train_remote.py --mode texts --file training_data.txt

  # 推論テスト (inference API)
  python train_remote.py --mode inference --prompt "日本の首都は"

  # 全部まとめて (qa + dataset)
  python train_remote.py --mode all
"""
import argparse
import json
import os
import sys
import time
import requests

ENDPOINT_URL = "https://vvcci2ps4y3wfx7m.us-east4.gcp.endpoints.huggingface.cloud"

# 高品質QAペアデータ (train_qa API用)
QA_PAIRS = [
    {"question": "日本の首都はどこですか？", "answer": "日本の首都は東京です。東京は政治、経済、文化の中心地であり、世界最大級の都市圏を形成しています。"},
    {"question": "富士山の高さは何メートルですか？", "answer": "富士山の高さは3776メートルです。日本で最も高い山であり、世界遺産に登録されています。"},
    {"question": "太陽系で一番大きい惑星は何ですか？", "answer": "木星が太陽系で最も大きい惑星です。直径は約14万キロメートルで、地球の約11倍の大きさがあります。"},
    {"question": "水の化学式は何ですか？", "answer": "水の化学式はH2Oです。水素原子2つと酸素原子1つから構成されています。"},
    {"question": "光の速さはどれくらいですか？", "answer": "光の速さは秒速約30万キロメートル（299,792,458 m/s）です。宇宙で最も速い速度です。"},
    {"question": "日本で一番長い川は何ですか？", "answer": "信濃川が日本で最も長い川で、全長367キロメートルです。新潟県と長野県を流れています。"},
    {"question": "地球の年齢はどれくらいですか？", "answer": "地球の年齢は約46億年です。太陽系の形成とほぼ同時期に誕生しました。"},
    {"question": "人間の体で一番大きい臓器は何ですか？", "answer": "皮膚が人間の体で最も大きい臓器です。体全体を覆い、外部からの保護や体温調節を行います。"},
    {"question": "明治維新は何年に起きましたか？", "answer": "明治維新は1868年に起きました。江戸幕府が倒れ、天皇中心の近代国家が成立しました。"},
    {"question": "第二次世界大戦はいつ終わりましたか？", "answer": "第二次世界大戦は1945年に終わりました。日本は8月15日に降伏を発表しました。"},
    {"question": "源頼朝は何をした人ですか？", "answer": "源頼朝は鎌倉幕府を開いた人物です。1185年に壇ノ浦の戦いで平氏を滅ぼし、1192年に征夷大将軍に任命されました。"},
    {"question": "江戸時代はいつからいつまでですか？", "answer": "江戸時代は1603年から1868年までの約265年間です。徳川家康が江戸幕府を開きました。"},
    {"question": "DNAとは何ですか？", "answer": "DNAはデオキシリボ核酸の略で、遺伝情報を保持する分子です。二重らせん構造を持ち、すべての生物の設計図となっています。"},
    {"question": "重力とは何ですか？", "answer": "重力とは、質量を持つ物体同士が引き合う力です。ニュートンの万有引力の法則で説明され、地球上では約9.8m/s²の加速度を生じます。"},
    {"question": "光合成とは何ですか？", "answer": "光合成とは、植物が光エネルギーを使って二酸化炭素と水からブドウ糖と酸素を作り出す反応です。"},
    {"question": "原子とは何ですか？", "answer": "原子は物質を構成する最小の単位で、陽子と中性子からなる原子核と、その周りを回る電子で構成されています。"},
    {"question": "俳句とは何ですか？", "answer": "俳句は五七五の17音で詠む日本の伝統的な短詩です。季語を含むことが基本的なルールです。松尾芭蕉が有名です。"},
    {"question": "歌舞伎とは何ですか？", "answer": "歌舞伎は日本の伝統的な演劇で、舞踊と音楽を組み合わせた総合芸術です。ユネスコ無形文化遺産に登録されています。"},
    {"question": "茶道について教えてください。", "answer": "茶道は日本の伝統文化で、抹茶を点てて客人をもてなす儀式です。千利休が大成させ、わび・さびの精神を大切にしています。"},
    {"question": "日本の国花は何ですか？", "answer": "日本の国花は桜と菊です。桜は春の象徴として花見文化があり、菊は皇室の紋章として使われています。"},
    {"question": "日本で一番大きい湖は何ですか？", "answer": "琵琶湖が日本で最も大きい湖です。滋賀県にあり、面積は約670平方キロメートルです。"},
    {"question": "北海道の県庁所在地はどこですか？", "answer": "北海道の道庁所在地は札幌市です。人口約197万人で、北海道の政治・経済の中心です。"},
    {"question": "円周率とは何ですか？", "answer": "円周率は円の周の長さと直径の比で、約3.14159です。πという記号で表され、無理数です。"},
    {"question": "三角形の内角の和は何度ですか？", "answer": "三角形の内角の和は180度です。これはユークリッド幾何学の基本定理の一つです。"},
    {"question": "素数とは何ですか？", "answer": "素数とは、1とその数自身でしか割り切れない2以上の自然数です。2、3、5、7、11などが素数です。"},
    {"question": "ピタゴラスの定理とは何ですか？", "answer": "ピタゴラスの定理は、直角三角形において斜辺の2乗が他の2辺の2乗の和に等しいという定理です。a²+b²=c²で表されます。"},
    {"question": "人工知能とは何ですか？", "answer": "人工知能（AI）は、人間の知的能力をコンピュータで実現する技術です。機械学習やディープラーニングなどの手法があります。"},
    {"question": "量子コンピュータとは何ですか？", "answer": "量子コンピュータは量子力学の原理（重ね合わせやエンタングルメント）を利用して計算を行うコンピュータです。従来のコンピュータでは困難な問題を高速に解ける可能性があります。"},
    {"question": "インターネットとは何ですか？", "answer": "インターネットは世界中のコンピュータネットワークを相互に接続した通信基盤です。1969年のARPANETが起源で、現在は数十億の機器が接続されています。"},
    {"question": "プログラミングとは何ですか？", "answer": "プログラミングとは、コンピュータに実行させたい処理をプログラミング言語で記述することです。Python、Java、C++などの言語があります。"},
    {"question": "健康のために大切なことは何ですか？", "answer": "健康のためには、バランスの良い食事、適度な運動、十分な睡眠が大切です。また、ストレス管理と定期的な健康診断も重要です。"},
    {"question": "環境問題について教えてください。", "answer": "環境問題には地球温暖化、大気汚染、海洋プラスチック汚染、森林破壊などがあります。CO2排出削減やリサイクルなどの取り組みが進められています。"},
    {"question": "ChatGPTとは何ですか？", "answer": "ChatGPTはOpenAIが開発した大規模言語モデルベースのAIチャットボットです。自然言語で対話でき、質問応答や文章作成など幅広いタスクに対応します。"},
    {"question": "東京タワーの高さは何メートルですか？", "answer": "東京タワーの高さは333メートルです。1958年に完成し、東京のシンボルとして親しまれています。"},
    {"question": "ノーベル賞とは何ですか？", "answer": "ノーベル賞はアルフレッド・ノーベルの遺言によって創設された国際的な賞です。物理学、化学、医学、文学、平和、経済学の6部門があります。"},
    {"question": "月の直径はどれくらいですか？", "answer": "月の直径は約3474キロメートルで、地球の約4分の1の大きさです。"},
    {"question": "酸素の元素記号は何ですか？", "answer": "酸素の元素記号はOです。原子番号8で、地球の大気の約21%を占めています。"},
    {"question": "日本の国歌は何ですか？", "answer": "日本の国歌は「君が代」です。歌詞は古今和歌集に収録された和歌に基づいています。"},
    {"question": "睡眠はなぜ大切ですか？", "answer": "睡眠は体の回復、記憶の整理、免疫力の維持に不可欠です。成人は一般的に7〜9時間の睡眠が推奨されています。"},
    {"question": "読書のメリットは何ですか？", "answer": "読書には知識の拡大、語彙力の向上、想像力の発達、ストレス軽減などのメリットがあります。集中力や思考力も鍛えられます。"},
]


# ── API helpers ─────────────────────────────────────────────

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
                print(f"    Model loading, waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"    Timeout (attempt {attempt+1}/{retries})")
        except Exception as e:
            print(f"    Error: {e}")
        if attempt < retries - 1:
            wait = 2 ** (attempt + 1)
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)

    return {"error": "All retries failed"}


# ── Action: status ──────────────────────────────────────────

def check_status(endpoint_url):
    """Query endpoint status."""
    print(f"\n  Checking status...")
    result = _post(endpoint_url, {
        "inputs": "__status__",
        "parameters": {"action": "status"}
    }, timeout=30)
    return result


# ── Action: inference ───────────────────────────────────────

def run_inference(endpoint_url, prompt, max_new_tokens=100, temperature=0.7):
    """Run text generation inference."""
    result = _post(endpoint_url, {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.3,
        }
    }, timeout=60)
    return result


# ── Action: train_qa ────────────────────────────────────────

def train_qa(endpoint_url, qa_pairs, epochs=6, repeat=3, batch_size=4):
    """Train with QA pairs using the train_qa API."""
    print(f"\n{'='*60}")
    print(f"  Remote Training: train_qa")
    print(f"  QA Pairs: {len(qa_pairs)}")
    print(f"  Repeat: {repeat}, Epochs: {epochs}")
    print(f"{'='*60}")

    # Split QA pairs into chunks to avoid request size limits
    chunk_size = max(len(qa_pairs) // 2, 1)
    chunks = [qa_pairs[i:i+chunk_size] for i in range(0, len(qa_pairs), chunk_size)]

    results = []
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i+1}/{len(chunks)} ({len(chunk)} QA pairs)...")
        result = _post(endpoint_url, {
            "inputs": "__train_qa__",
            "parameters": {
                "action": "train_qa",
                "qa_pairs": chunk,
                "epochs": epochs,
                "lr": 3e-5,
                "batch_size": batch_size,
                "grad_accum_steps": 4,
                "repeat": repeat,
            }
        })
        print(f"    Result: {json.dumps(result, ensure_ascii=False)[:200]}")
        results.append(result)

    return results


# ── Action: train ───────────────────────────────────────────

def train_dataset(endpoint_url, dataset_id, text_column="text",
                   max_samples=200, epochs=5, lr=3e-5):
    """Train with a HF dataset (loaded on endpoint side)."""
    print(f"\n{'='*60}")
    print(f"  Remote Training: train (dataset)")
    print(f"  Dataset: {dataset_id}")
    print(f"  Max Samples: {max_samples}, Epochs: {epochs}")
    print(f"{'='*60}")

    result = _post(endpoint_url, {
        "inputs": "__train__",
        "parameters": {
            "action": "train",
            "dataset_id": dataset_id,
            "text_column": text_column,
            "max_samples": max_samples,
            "epochs": epochs,
            "lr": lr,
        }
    })
    print(f"  Result: {json.dumps(result, ensure_ascii=False)[:300]}")
    return result


# ── Action: train_texts ─────────────────────────────────────

def train_texts(endpoint_url, texts, epochs=4, batch_size=4):
    """Train with raw text data sent directly."""
    print(f"\n{'='*60}")
    print(f"  Remote Training: train_texts")
    print(f"  Texts: {len(texts)}, Epochs: {epochs}")
    print(f"{'='*60}")

    chunk_size = 50
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]

    results = []
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i+1}/{len(chunks)} ({len(chunk)} texts)...")
        result = _post(endpoint_url, {
            "inputs": "__train_texts__",
            "parameters": {
                "action": "train_texts",
                "texts": chunk,
                "epochs": epochs,
                "lr": 3e-5,
                "batch_size": batch_size,
                "grad_accum_steps": 4,
                "warmup_steps": 30,
            }
        })
        print(f"    Result: {json.dumps(result, ensure_ascii=False)[:200]}")
        results.append(result)

    return results


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Remote training client for HF Inference Endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Actions:
  status     学習状況・モデル情報の確認
  inference  テキスト生成（推論テスト）
  qa         QA形式データでの学習 (train_qa API)
  dataset    一般データセットによる学習 (train API)
  texts      テキストファイルを送信して学習 (train_texts API)
  all        qa + dataset を順番に実行
        """
    )
    parser.add_argument("--endpoint", default=ENDPOINT_URL, help="Endpoint URL")
    parser.add_argument("--mode", choices=["status", "inference", "qa", "dataset", "texts", "all"],
                        default="all", help="Action mode")
    parser.add_argument("--prompt", default="日本の首都は", help="Inference prompt")
    parser.add_argument("--dataset_id", default="fujiki/japanese_alpaca_data",
                        help="HF dataset ID (for dataset mode)")
    parser.add_argument("--text_column", default="output", help="Text column name")
    parser.add_argument("--file", help="Path to text file (for texts mode)")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per chunk")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples from dataset")
    parser.add_argument("--repeat", type=int, default=3, help="QA data repeat count")
    args = parser.parse_args()

    print("=" * 60)
    print("  Remote Training Client v2")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Mode: {args.mode}")
    print("=" * 60)

    # Check endpoint health
    try:
        resp = requests.get(f"{args.endpoint}/health", timeout=10)
        if resp.status_code == 200:
            print("  Endpoint: OK")
        else:
            print(f"  Warning: Health check returned {resp.status_code}")
    except Exception as e:
        print(f"  Error: Cannot reach endpoint: {e}")
        return

    # ── Dispatch ────────────────────────────────────────────

    if args.mode == "status":
        result = check_status(args.endpoint)
        print(f"\n  Status: {json.dumps(result, indent=2, ensure_ascii=False)}")

    elif args.mode == "inference":
        result = run_inference(args.endpoint, args.prompt)
        print(f"\n  Prompt: {args.prompt}")
        print(f"  Output: {result.get('generated_text', result)}")

    elif args.mode == "qa":
        results = train_qa(args.endpoint, QA_PAIRS, epochs=args.epochs, repeat=args.repeat)
        print(f"\n  QA Training: {len(results)} chunks completed")

    elif args.mode == "dataset":
        result = train_dataset(
            args.endpoint, args.dataset_id,
            text_column=args.text_column,
            max_samples=args.max_samples,
            epochs=args.epochs,
        )

    elif args.mode == "texts":
        if not args.file:
            print("  Error: --file required for texts mode")
            return
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip() and len(line.strip()) > 10]
        print(f"  Loaded {len(texts)} texts from {args.file}")
        results = train_texts(args.endpoint, texts, epochs=args.epochs)

    elif args.mode == "all":
        # Step 1: Status check
        print("\n[1/3] Status check...")
        status = check_status(args.endpoint)
        print(f"  {json.dumps(status, ensure_ascii=False)[:200]}")

        # Step 2: QA training
        print("\n[2/3] QA training...")
        qa_results = train_qa(args.endpoint, QA_PAIRS, epochs=args.epochs, repeat=args.repeat)
        print(f"  QA: {len(qa_results)} chunks")

        # Step 3: Dataset training
        print("\n[3/3] Dataset training...")
        ds_result = train_dataset(
            args.endpoint, args.dataset_id,
            text_column=args.text_column,
            max_samples=args.max_samples,
            epochs=args.epochs,
        )

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
