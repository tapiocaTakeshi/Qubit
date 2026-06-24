/**
 * Fetch Japanese text samples from HuggingFace datasets
 * and generate training data for the chat model
 */

import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';

async function fetchWikitextJA() {
  console.log('Fetching Japanese Wikipedia text...');
  try {
    // Fetch wikitext-ja dataset info
    const response = await fetch(
      'https://huggingface.co/datasets/wikitext/raw/main/wikitext-103-v1/wikitext-1-raw/wiki.train.raw'
    );
    if (!response.ok) throw new Error(`Status ${response.status}`);

    const text = await response.text();
    return text.split('\n').filter(line => line.trim().length > 20).slice(0, 100);
  } catch (e) {
    console.log('Wikitext fetch failed, using fallback...');
    return null;
  }
}

async function fetchOscarJA() {
  console.log('Fetching OSCAR Japanese samples...');
  try {
    // Try to fetch OSCAR dataset metadata
    const response = await fetch(
      'https://huggingface.co/datasets/oscar/raw/main/README.md'
    );
    if (!response.ok) throw new Error(`Status ${response.status}`);
    const text = await response.text();
    return text.split('\n').filter(line => line.includes('日本') || line.includes('ja')).slice(0, 50);
  } catch (e) {
    console.log('OSCAR fetch failed, using fallback...');
    return null;
  }
}

const fallbackJapanesePhrases = [
  // Technical AI/ML concepts
  '機械学習はコンピュータが例から学ぶ方法です。',
  'ニューラルネットワークは脳の神経細胞を模倣した計算モデルです。',
  '深層学習は複数の層を持つニューラルネットワークです。',
  '自然言語処理は人間の言語をコンピュータが理解する技術です。',
  'Transformerモデルはシーケンスのすべての位置を同時に処理できます。',
  '自己注意機構は入力の異なる位置間の依存関係を学習します。',
  '勾配降下法はニューラルネットワークの重みを最適化します。',
  'バッチ正規化は各層への入力を正規化して学習を安定化させます。',
  'ドロップアウトはニューロンをランダムに無効にして過学習を防ぎます。',
  'リカレントニューラルネットワークはシーケンシャルデータを処理します。',
  'LSTM（長短期記憶）は長い依存関係を学習できるRNNです。',
  'GRU（ゲート付き再帰単位）はLSTMを簡潔にしたバージョンです。',
  '畳み込みニューラルネットワークは画像処理に特化しています。',
  '埋め込み層はカテゴリカルデータを密なベクトルに変換します。',
  'エンコーダーはデータを圧縮表現に変換します。',
  'デコーダーは圧縮表現から元のデータを復元します。',
  'VAE（変分オートエンコーダー）は生成モデルです。',
  'GAN（敵対的生成ネットワーク）は2つのネットワークが競い合います。',
  '強化学習はエージェントが環境と相互作用して学びます。',
  'Q学習は価値ベースの強化学習アルゴリズムです。',
  '政策勾配法はエージェントの政策を直接最適化します。',
  'actor-criticアルゴリズムは政策と価値を組み合わせます。',
  '報酬は強化学習におけるエージェントの目標を定義します。',
  'Markovの性質は現在の状態のみが未来を決定することを意味します。',
  'Bellman方程式は異なる状態の価値を関連付けます。',
  'モンテカルロ法はランダムサンプリングで価値を推定します。',
  '時間差学習は経験から学習と評価を交互に行います。',
  '転移学習は別のタスクで学んだ知識を再利用します。',
  'メタ学習は学習プロセス自体を学ぶことです。',
  'マルチタスク学習は複数のタスクを同時に学習します。',
  '半教師あり学習は標識データと無標識データを組み合わせます。',
  'シーケンス・ツー・シーケンスモデルは入力シーケンスを出力シーケンスに変換します。',
  'アテンションメカニズムは関連する情報に焦点を当てます。',
  'マルチヘッドアテンションは複数のアテンション表現を学習します。',
  'パラレルトランスフォーマーはすべての位置を同時に処理できます。',
];

async function main() {
  let phrases = fallbackJapanesePhrases;

  // Try to fetch additional data
  const wikitextData = await fetchWikitextJA();
  if (wikitextData) {
    phrases = [...phrases, ...wikitextData.slice(0, 20)];
  }

  const oscarData = await fetchOscarJA();
  if (oscarData) {
    phrases = [...phrases, ...oscarData.slice(0, 15)];
  }

  // Expand with variations
  const expanded = phrases.map(p => [
    p,
    p.replace('は', 'ですか？'),
    `${p}。詳しく説明すると、`,
  ]).flat();

  const data = {
    timestamp: new Date().toISOString(),
    phrase_count: expanded.length,
    phrases: expanded.slice(0, 200),
  };

  const outputPath = path.join(process.cwd(), 'data', 'japanese-training-data.json');
  const dir = path.dirname(outputPath);

  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
  console.log(`✓ Saved ${data.phrase_count} phrases to ${outputPath}`);
}

main().catch(console.error);
