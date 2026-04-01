"""Train SentencePiece tokenizer for NeuroQuantum model.

Downloads Japanese QA datasets and trains a Unigram SP model with
special tokens matching NeuroQ convention:
  pad=0, unk=1, bos=2, eos=3, bof=4, eof=5
"""
import os
import sys
import tempfile

import sentencepiece as spm


def main():
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 20000

    print(f"Training SP tokenizer: vocab_size={vocab_size}, max_samples={max_samples}")

    # Collect training texts
    texts = []
    try:
        from datasets import load_dataset

        for ds_id in [
            "fujiki/japanese_alpaca_data",
            "FreedomIntelligence/alpaca-gpt4-japanese",
        ]:
            try:
                ds = load_dataset(ds_id, split=f"train[:{max_samples}]")
                for row in ds:
                    for key in ["instruction", "output", "input"]:
                        t = row.get(key, "").strip()
                        if t and len(t) > 5:
                            texts.append(t)
                print(f"  {ds_id}: loaded, total so far {len(texts)}")
            except Exception as e:
                print(f"  {ds_id}: skip ({e})")
    except ImportError:
        print("  datasets not available, using built-in corpus only")

    # Built-in corpus (always included)
    builtin = [
        "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。東京は関東地方に位置し、政治・経済・文化の中心地です。",
        "質問: 量子コンピュータとは何ですか？\n回答: 量子コンピュータは、量子力学の原理を利用して計算を行うコンピュータです。",
        "質問: Pythonでリストをソートする方法を教えてください。\n回答: sort()メソッドまたはsorted()関数を使用します。",
        "質問: 機械学習とは何ですか？\n回答: 機械学習は、データからパターンを学習し、予測や判断を行う人工知能の一分野です。",
        "質問: データベースとは何ですか？\n回答: データベースは、構造化されたデータの集合です。SQLを使ってデータの操作を行います。",
        "人工知能は急速に発展しており、自然言語処理、画像認識、音声合成など多くの分野で活用されています。",
        "深層学習は多層のニューラルネットワークを使用して、複雑なパターンを学習する手法です。",
        "トランスフォーマーアーキテクチャは、自己注意機構を使用して入力データの関連性を効率的に処理します。",
        "GPUは並列計算に優れたプロセッサで、深層学習の学習を大幅に高速化します。",
        "Dockerはコンテナ技術を用いた仮想化プラットフォームで、アプリケーションの移植性を高めます。",
        "Python is a versatile programming language used for web development, data science, and AI.",
        "Machine learning models require large datasets for training and validation.",
        "The transformer architecture revolutionized natural language processing with self-attention.",
    ]
    texts.extend(builtin * 10)

    if len(texts) < 100:
        print(f"ERROR: Only {len(texts)} texts, need at least 100")
        sys.exit(1)

    print(f"Total training texts: {len(texts)}")

    # Write temp file
    train_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    for t in texts:
        train_file.write(t.replace("\n", " ").strip() + "\n")
    train_file.close()

    # Train
    for prefix in ["neuroq_tokenizer", "neuroq_tokenizer_8k"]:
        model_path = os.path.join(output_dir, prefix)
        spm.SentencePieceTrainer.Train(
            input=train_file.name,
            model_prefix=model_path,
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            user_defined_symbols="<bof>,<eof>",
            num_threads=4,
            max_sentence_length=16384,
            input_sentence_size=200000,
        )

        sp = spm.SentencePieceProcessor()
        sp.Load(f"{model_path}.model")
        print(f"  {prefix}: vocab_size={sp.GetPieceSize()}")

        test = "質問: 日本の首都はどこですか？"
        ids = sp.EncodeAsIds(test)
        print(f"    test encode ({len(ids)} tokens): {ids[:15]}")

    os.unlink(train_file.name)
    print("Done!")


if __name__ == "__main__":
    main()
