#!/usr/bin/env python3
"""
リモート学習クライアント - HF Inference Endpointのtrain APIを叩いて学習させる

使い方:
  # デフォルト（QAデータ + Wikipedia）でリモート学習
  python train_remote.py

  # データセットを指定してリモート学習
  python train_remote.py --mode dataset --dataset_id fujiki/japanese_alpaca_data

  # ローカルのテキストファイルを送信して学習
  python train_remote.py --mode texts --file training_data.txt

  # QAデータを直接送信して学習
  python train_remote.py --mode qa
"""
import argparse
import json
import os
import sys
import time
import requests

ENDPOINT_URL = "https://vvcci2ps4y3wfx7m.us-east4.gcp.endpoints.huggingface.cloud"

# 高品質QAデータ
QA_DATA = [
    "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。東京は政治、経済、文化の中心地であり、世界最大級の都市圏を形成しています。",
    "質問: 富士山の高さは何メートルですか？\n回答: 富士山の高さは3776メートルです。日本で最も高い山であり、世界遺産に登録されています。",
    "質問: 太陽系で一番大きい惑星は何ですか？\n回答: 木星が太陽系で最も大きい惑星です。直径は約14万キロメートルで、地球の約11倍の大きさがあります。",
    "質問: 水の化学式は何ですか？\n回答: 水の化学式はH2Oです。水素原子2つと酸素原子1つから構成されています。",
    "質問: 光の速さはどれくらいですか？\n回答: 光の速さは秒速約30万キロメートル（299,792,458 m/s）です。宇宙で最も速い速度です。",
    "質問: 日本で一番長い川は何ですか？\n回答: 信濃川が日本で最も長い川で、全長367キロメートルです。新潟県と長野県を流れています。",
    "質問: 地球の年齢はどれくらいですか？\n回答: 地球の年齢は約46億年です。太陽系の形成とほぼ同時期に誕生しました。",
    "質問: 人間の体で一番大きい臓器は何ですか？\n回答: 皮膚が人間の体で最も大きい臓器です。体全体を覆い、外部からの保護や体温調節を行います。",
    "質問: 明治維新は何年に起きましたか？\n回答: 明治維新は1868年に起きました。江戸幕府が倒れ、天皇中心の近代国家が成立しました。",
    "質問: 第二次世界大戦はいつ終わりましたか？\n回答: 第二次世界大戦は1945年に終わりました。日本は8月15日に降伏を発表しました。",
    "質問: 源頼朝は何をした人ですか？\n回答: 源頼朝は鎌倉幕府を開いた人物です。1185年に壇ノ浦の戦いで平氏を滅ぼし、1192年に征夷大将軍に任命されました。",
    "質問: 江戸時代はいつからいつまでですか？\n回答: 江戸時代は1603年から1868年までの約265年間です。徳川家康が江戸幕府を開きました。",
    "質問: DNAとは何ですか？\n回答: DNAはデオキシリボ核酸の略で、遺伝情報を保持する分子です。二重らせん構造を持ち、すべての生物の設計図となっています。",
    "質問: 重力とは何ですか？\n回答: 重力とは、質量を持つ物体同士が引き合う力です。ニュートンの万有引力の法則で説明され、地球上では約9.8m/s²の加速度を生じます。",
    "質問: 光合成とは何ですか？\n回答: 光合成とは、植物が光エネルギーを使って二酸化炭素と水からブドウ糖と酸素を作り出す反応です。",
    "質問: 原子とは何ですか？\n回答: 原子は物質を構成する最小の単位で、陽子と中性子からなる原子核と、その周りを回る電子で構成されています。",
    "質問: 俳句とは何ですか？\n回答: 俳句は五七五の17音で詠む日本の伝統的な短詩です。季語を含むことが基本的なルールです。松尾芭蕉が有名です。",
    "質問: 歌舞伎とは何ですか？\n回答: 歌舞伎は日本の伝統的な演劇で、舞踊と音楽を組み合わせた総合芸術です。ユネスコ無形文化遺産に登録されています。",
    "質問: 茶道について教えてください。\n回答: 茶道は日本の伝統文化で、抹茶を点てて客人をもてなす儀式です。千利休が大成させ、わび・さびの精神を大切にしています。",
    "質問: 日本の国花は何ですか？\n回答: 日本の国花は桜と菊です。桜は春の象徴として花見文化があり、菊は皇室の紋章として使われています。",
    "質問: 日本で一番大きい湖は何ですか？\n回答: 琵琶湖が日本で最も大きい湖です。滋賀県にあり、面積は約670平方キロメートルです。",
    "質問: 北海道の県庁所在地はどこですか？\n回答: 北海道の道庁所在地は札幌市です。人口約197万人で、北海道の政治・経済の中心です。",
    "質問: 円周率とは何ですか？\n回答: 円周率は円の周の長さと直径の比で、約3.14159です。πという記号で表され、無理数です。",
    "質問: 三角形の内角の和は何度ですか？\n回答: 三角形の内角の和は180度です。これはユークリッド幾何学の基本定理の一つです。",
    "質問: 素数とは何ですか？\n回答: 素数とは、1とその数自身でしか割り切れない2以上の自然数です。2、3、5、7、11などが素数です。",
    "質問: ピタゴラスの定理とは何ですか？\n回答: ピタゴラスの定理は、直角三角形において斜辺の2乗が他の2辺の2乗の和に等しいという定理です。a²+b²=c²で表されます。",
    "質問: 人工知能とは何ですか？\n回答: 人工知能（AI）は、人間の知的能力をコンピュータで実現する技術です。機械学習やディープラーニングなどの手法があります。",
    "質問: 量子コンピュータとは何ですか？\n回答: 量子コンピュータは量子力学の原理（重ね合わせやエンタングルメント）を利用して計算を行うコンピュータです。従来のコンピュータでは困難な問題を高速に解ける可能性があります。",
    "質問: インターネットとは何ですか？\n回答: インターネットは世界中のコンピュータネットワークを相互に接続した通信基盤です。1969年のARPANETが起源で、現在は数十億の機器が接続されています。",
    "質問: プログラミングとは何ですか？\n回答: プログラミングとは、コンピュータに実行させたい処理をプログラミング言語で記述することです。Python、Java、C++などの言語があります。",
    "質問: 健康のために大切なことは何ですか？\n回答: 健康のためには、バランスの良い食事、適度な運動、十分な睡眠が大切です。また、ストレス管理と定期的な健康診断も重要です。",
    "質問: 環境問題について教えてください。\n回答: 環境問題には地球温暖化、大気汚染、海洋プラスチック汚染、森林破壊などがあります。CO2排出削減やリサイクルなどの取り組みが進められています。",
    "質問: ChatGPTとは何ですか？\n回答: ChatGPTはOpenAIが開発した大規模言語モデルベースのAIチャットボットです。自然言語で対話でき、質問応答や文章作成など幅広いタスクに対応します。",
    "質問: 東京タワーの高さは何メートルですか？\n回答: 東京タワーの高さは333メートルです。1958年に完成し、東京のシンボルとして親しまれています。",
    "質問: ノーベル賞とは何ですか？\n回答: ノーベル賞はアルフレッド・ノーベルの遺言によって創設された国際的な賞です。物理学、化学、医学、文学、平和、経済学の6部門があります。",
    "質問: 月の直径はどれくらいですか？\n回答: 月の直径は約3474キロメートルで、地球の約4分の1の大きさです。",
    "質問: 酸素の元素記号は何ですか？\n回答: 酸素の元素記号はOです。原子番号8で、地球の大気の約21%を占めています。",
    "質問: 日本の国歌は何ですか？\n回答: 日本の国歌は「君が代」です。歌詞は古今和歌集に収録された和歌に基づいています。",
    "質問: 睡眠はなぜ大切ですか？\n回答: 睡眠は体の回復、記憶の整理、免疫力の維持に不可欠です。成人は一般的に7〜9時間の睡眠が推奨されています。",
    "質問: 読書のメリットは何ですか？\n回答: 読書には知識の拡大、語彙力の向上、想像力の発達、ストレス軽減などのメリットがあります。集中力や思考力も鍛えられます。",
]


def send_train_request(endpoint_url, texts, epochs=4, lr=3e-5, batch_size=4,
                        grad_accum=4, chunk_idx=0, total_chunks=1, retries=3):
    """Send training texts to endpoint."""
    payload = {
        "inputs": "__train_texts__",
        "parameters": {
            "action": "train_texts",
            "texts": texts,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum,
            "warmup_steps": 30,
        }
    }

    for attempt in range(retries):
        try:
            print(f"  Sending chunk {chunk_idx+1}/{total_chunks} ({len(texts)} texts)...")
            resp = requests.post(
                endpoint_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=600,  # 10 min timeout for training
            )
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list):
                    result = result[0]
                return result
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code == 503:
                    wait = 10 * (attempt + 1)
                    print(f"  Model loading, waiting {wait}s...")
                    time.sleep(wait)
                    continue
        except requests.exceptions.Timeout:
            print(f"  Timeout (attempt {attempt+1}/{retries})")
        except Exception as e:
            print(f"  Error: {e}")
        if attempt < retries - 1:
            wait = 2 ** (attempt + 1)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    return {"error": "All retries failed"}


def send_dataset_train(endpoint_url, dataset_id, text_column="text",
                        max_samples=200, epochs=5, lr=3e-5):
    """Send dataset training request (uses HF datasets on endpoint side)."""
    payload = {
        "inputs": "",
        "parameters": {
            "action": "train",
            "dataset_id": dataset_id,
            "text_column": text_column,
            "max_samples": max_samples,
            "epochs": epochs,
            "lr": lr,
        }
    }
    print(f"  Training with dataset: {dataset_id}")
    try:
        resp = requests.post(
            endpoint_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=600,
        )
        if resp.status_code == 200:
            result = resp.json()
            if isinstance(result, list):
                result = result[0]
            return result
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def train_qa(endpoint_url, epochs=6, num_repeats=3):
    """Train with QA data by sending texts directly to the endpoint."""
    print(f"\n{'='*60}")
    print(f"  Remote Training: QA Data")
    print(f"  Endpoint: {endpoint_url}")
    print(f"  QA Samples: {len(QA_DATA)}")
    print(f"  Repeats: {num_repeats}, Epochs per chunk: {epochs}")
    print(f"{'='*60}")

    # Send QA data in chunks with repeats for better learning
    all_texts = QA_DATA * num_repeats
    chunk_size = max(len(all_texts) // 4, 1)
    chunks = [all_texts[i:i+chunk_size] for i in range(0, len(all_texts), chunk_size)]

    results = []
    for i, chunk in enumerate(chunks):
        result = send_train_request(
            endpoint_url, chunk,
            epochs=epochs, lr=3e-5, batch_size=4, grad_accum=4,
            chunk_idx=i, total_chunks=len(chunks)
        )
        print(f"  Chunk {i+1}: {result}")
        results.append(result)

    return results


def train_from_file(endpoint_url, filepath, epochs=4):
    """Train with texts from a local file."""
    print(f"\n  Loading texts from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip() and len(line.strip()) > 10]

    print(f"  Loaded {len(texts)} texts")

    chunk_size = 50
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]

    results = []
    for i, chunk in enumerate(chunks):
        result = send_train_request(
            endpoint_url, chunk,
            epochs=epochs, chunk_idx=i, total_chunks=len(chunks)
        )
        print(f"  Chunk {i+1}: {result}")
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Remote training client for HF Inference Endpoint")
    parser.add_argument("--endpoint", default=ENDPOINT_URL, help="Endpoint URL")
    parser.add_argument("--mode", choices=["qa", "dataset", "texts", "all"], default="all",
                        help="Training mode")
    parser.add_argument("--dataset_id", default="fujiki/japanese_alpaca_data",
                        help="HF dataset ID (for dataset mode)")
    parser.add_argument("--text_column", default="output", help="Text column name")
    parser.add_argument("--file", help="Path to text file (for texts mode)")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per chunk")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples from dataset")
    parser.add_argument("--repeats", type=int, default=3, help="QA data repeat count")
    args = parser.parse_args()

    print("=" * 60)
    print("  Remote Training Client")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Mode: {args.mode}")
    print("=" * 60)

    # Check endpoint health
    try:
        resp = requests.get(f"{args.endpoint}/health", timeout=10)
        if resp.status_code == 200:
            print("  Endpoint: OK")
        else:
            print(f"  Warning: Endpoint health check returned {resp.status_code}")
    except Exception as e:
        print(f"  Warning: Cannot reach endpoint: {e}")
        return

    if args.mode == "qa" or args.mode == "all":
        results = train_qa(args.endpoint, epochs=args.epochs, num_repeats=args.repeats)
        print(f"\n  QA Training Results: {len(results)} chunks completed")

    if args.mode == "dataset" or args.mode == "all":
        result = send_dataset_train(
            args.endpoint, args.dataset_id,
            text_column=args.text_column,
            max_samples=args.max_samples,
            epochs=args.epochs
        )
        print(f"\n  Dataset Training Result: {result}")

    if args.mode == "texts":
        if not args.file:
            print("  Error: --file required for texts mode")
            return
        results = train_from_file(args.endpoint, args.file, epochs=args.epochs)
        print(f"\n  Text Training Results: {len(results)} chunks completed")

    print("\n" + "=" * 60)
    print("  Remote Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
