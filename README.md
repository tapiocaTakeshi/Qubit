---
title: NeuroQ
emoji: 🧠⚛️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: true
license: mit
tags:
  - japanese
  - qbnn
  - quantum
  - transformer
  - chat
short_description: "脳型量子ビットネットワーク(QBNN)による日本語生成AI"
---

# 🧠⚛️ NeuroQ - 脳型量子ビットネットワーク チャットボット

**QBNN (Quantum Bit Neural Network) Transformer** による日本語対話AI

## 特徴

- ⚛️ **量子もつれ層 (QBNN Layer)**: Bloch球マッピングと動的エンタングルメントで独自の表現学習
- 🧠 **脳型アーキテクチャ**: 層間エンタングルメントを持つTransformerモデル
- 🇯🇵 **日本語3段階学習**: 事前学習 → SFT(指示チューニング) → DPO(選好学習)
- 🔤 **SentencePiece**: 8000語彙の日本語サブワードトークナイザー

## アーキテクチャ

```
入力 → Embedding → [QBNN-Transformer Block × N] → LM Head → 出力
                     ├── QBNN-Attention (量子もつれ補正付き)
                     ├── Feed-Forward Network
                     └── QBNN Layer (Bloch球 + エンタングルメント)
```

## 使い方

テキストボックスに日本語で質問や指示を入力してください。
パラメータ（Temperature、Top-K/P等）を調整して出力をコントロールできます。

## リンク

- [📂 GitHub](https://github.com/tapiocaTakeshi/NeuroQ)
