#!/usr/bin/env python3
"""
分割学習 (Split Learning) モジュール

モデルを「カットレイヤー」で分割し、クライアント側（下位層）と
サーバー側（上位層）で分散学習を行う。

アーキテクチャ:
    ┌─────────────────────────────────────┐
    │  クライアント側 (SplitClient)         │
    │  ├─ トークン埋め込み層               │
    │  ├─ 位置埋め込み層                   │
    │  ├─ ドロップアウト                   │
    │  └─ Transformerブロック × cut_layer  │
    └──────────────┬──────────────────────┘
                   │ 中間活性化 (smashed data)
                   ▼
    ┌─────────────────────────────────────┐
    │  サーバー側 (SplitServer)            │
    │  ├─ Transformerブロック × (N-cut)    │
    │  ├─ 最後のLayerNorm                 │
    │  └─ 線形出力層                       │
    └─────────────────────────────────────┘

プライバシー:
    - クライアントは生データを保持（サーバーには送らない）
    - 中間活性化（smashed data）のみがネットワークを通過
    - サーバーは逆伝播の勾配のみクライアントに返す

通信プロトコル:
    TCP ソケット + pickle シリアライゼーション
    1. クライアント → サーバー: 中間活性化テンソル + ラベル
    2. サーバー → クライアント: カットレイヤーでの勾配テンソル + loss値

使い方:
    # サーバー起動
    python split_learning.py --role server --host 0.0.0.0 --port 9000

    # クライアント起動
    python split_learning.py --role client --server_host 192.168.1.100 --server_port 9000
"""

import os
import sys
import io
import socket
import struct
import pickle
import logging
import time
import threading
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import (
    NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer,
    QBNNTransformerBlock, migrate_legacy_state_dict,
)

logger = logging.getLogger(__name__)


# ========================================
# メッセージ送受信ユーティリティ
# ========================================

def send_message(sock: socket.socket, data: Any) -> None:
    """長さプレフィックス付きでpickleデータを送信する。"""
    buf = pickle.dumps(data)
    length = struct.pack("!Q", len(buf))
    sock.sendall(length + buf)


def recv_message(sock: socket.socket) -> Any:
    """長さプレフィックス付きpickleデータを受信する。"""
    raw_len = _recv_exact(sock, 8)
    if raw_len is None:
        return None
    length = struct.unpack("!Q", raw_len)[0]
    data_buf = _recv_exact(sock, length)
    if data_buf is None:
        return None
    return pickle.loads(data_buf)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """ソケットからちょうどnバイト受信する。"""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ========================================
# クライアント側モデル (下位層)
# ========================================

class SplitClient(nn.Module):
    """
    分割学習のクライアント側モデル。
    埋め込み層 + 最初のcut_layer個のTransformerブロックを保持する。
    """

    def __init__(self, model: NeuroQuantum, cut_layer: int):
        """
        Args:
            model: 完全なNeuroQuantumモデル
            cut_layer: カットレイヤーのインデックス（0-indexed、この数のブロックをクライアントが保持）
        """
        super().__init__()
        self.config = model.config
        self.cut_layer = cut_layer

        if cut_layer < 1 or cut_layer >= model.config.num_layers:
            raise ValueError(
                f"cut_layer must be in [1, {model.config.num_layers - 1}], got {cut_layer}"
            )

        # 埋め込み層
        self.token_embedding = model.token_embedding
        self.position_embedding = model.position_embedding
        self.embedding = model.embedding
        self.use_openai_embedding = model.use_openai_embedding
        self.use_google_embedding = model.use_google_embedding
        self.embedding_dropout = model.embedding_dropout

        # クライアント側のTransformerブロック
        self.transformer_blocks = nn.ModuleList(
            [model.transformer_blocks[i] for i in range(cut_layer)]
        )

    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        クライアント側フォワードパス。カットレイヤーまでの中間活性化を返す。

        Args:
            token_ids: (batch, seq) トークンID
            mask: Optional attention mask

        Returns:
            (batch, seq, embed_dim) 中間活性化テンソル（smashed data）
        """
        batch, seq = token_ids.shape
        token_ids = token_ids.clamp(0, self.config.vocab_size - 1)
        if seq > self.config.max_seq_len:
            token_ids = token_ids[:, :self.config.max_seq_len]
            seq = self.config.max_seq_len

        # 埋め込み
        use_external = (self.use_openai_embedding or self.use_google_embedding) and self.embedding is not None
        if use_external:
            hidden_states = self.embedding(token_ids, texts=None)
        else:
            token_embeds = self.token_embedding(token_ids)
            positions = torch.arange(seq, device=token_ids.device).unsqueeze(0).expand(batch, -1)
            positions = positions.clamp(0, self.config.max_seq_len - 1)
            pos_embeds = self.position_embedding(positions)
            hidden_states = token_embeds + pos_embeds

        hidden_states = self.embedding_dropout(hidden_states)

        # Causal Mask
        if mask is None:
            mask = torch.tril(torch.ones(seq, seq, device=token_ids.device)).unsqueeze(0).unsqueeze(0)

        # クライアント側Transformerブロック
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, mask)

        return hidden_states


# ========================================
# サーバー側モデル (上位層)
# ========================================

class SplitServer(nn.Module):
    """
    分割学習のサーバー側モデル。
    残りのTransformerブロック + 最終LayerNorm + 出力ヘッドを保持する。
    """

    def __init__(self, model: NeuroQuantum, cut_layer: int):
        """
        Args:
            model: 完全なNeuroQuantumモデル
            cut_layer: カットレイヤーのインデックス
        """
        super().__init__()
        self.config = model.config
        self.cut_layer = cut_layer

        if cut_layer < 1 or cut_layer >= model.config.num_layers:
            raise ValueError(
                f"cut_layer must be in [1, {model.config.num_layers - 1}], got {cut_layer}"
            )

        # サーバー側のTransformerブロック
        self.transformer_blocks = nn.ModuleList(
            [model.transformer_blocks[i] for i in range(cut_layer, model.config.num_layers)]
        )

        # 最終層
        self.final_norm = model.final_norm
        self.output_head = model.output_head

    def forward(
        self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        サーバー側フォワードパス。中間活性化からロジットを計算する。

        Args:
            hidden_states: (batch, seq, embed_dim) クライアントからの中間活性化
            mask: Optional attention mask

        Returns:
            (batch, seq, vocab_size) ロジット
        """
        seq = hidden_states.shape[1]

        if mask is None:
            mask = torch.tril(
                torch.ones(seq, seq, device=hidden_states.device)
            ).unsqueeze(0).unsqueeze(0)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, mask)

        hidden_states = self.final_norm(hidden_states)
        logits = self.output_head(hidden_states)
        return logits


# ========================================
# モデル分割ユーティリティ
# ========================================

def split_model(
    model: NeuroQuantum, cut_layer: int
) -> Tuple[SplitClient, SplitServer]:
    """
    NeuroQuantumモデルをクライアントとサーバーに分割する。

    Args:
        model: 完全なNeuroQuantumモデル
        cut_layer: カットレイヤーのインデックス (1 ~ num_layers-1)

    Returns:
        (client_model, server_model)
    """
    client = SplitClient(model, cut_layer)
    server = SplitServer(model, cut_layer)
    return client, server


def merge_split_models(
    client: SplitClient,
    server: SplitServer,
    config: NeuroQuantumConfig,
) -> NeuroQuantum:
    """
    分割されたクライアントとサーバーのモデルを統合して完全なモデルに戻す。

    Args:
        client: クライアント側モデル
        server: サーバー側モデル
        config: モデル設定

    Returns:
        統合されたNeuroQuantumモデル
    """
    full_model = NeuroQuantum(config=config)

    # 埋め込み層をコピー
    if client.token_embedding is not None:
        full_model.token_embedding.load_state_dict(client.token_embedding.state_dict())
    if client.position_embedding is not None:
        full_model.position_embedding.load_state_dict(client.position_embedding.state_dict())
    full_model.embedding_dropout.load_state_dict(client.embedding_dropout.state_dict())

    # クライアント側Transformerブロックをコピー
    for i, block in enumerate(client.transformer_blocks):
        full_model.transformer_blocks[i].load_state_dict(block.state_dict())

    # サーバー側Transformerブロックをコピー
    cut_layer = client.cut_layer
    for i, block in enumerate(server.transformer_blocks):
        full_model.transformer_blocks[cut_layer + i].load_state_dict(block.state_dict())

    # 最終層をコピー
    full_model.final_norm.load_state_dict(server.final_norm.state_dict())
    full_model.output_head.load_state_dict(server.output_head.state_dict())

    return full_model


# ========================================
# 分割学習トレーナー（ローカル・シミュレーション）
# ========================================

class SplitLearningTrainer:
    """
    分割学習トレーナー。ローカルでクライアント・サーバー間の
    分割学習をシミュレートする（同一マシン上で動作）。

    ネットワーク越しの分散学習を行う場合は、
    SplitLearningClient / SplitLearningServer を使用する。
    """

    def __init__(
        self,
        model: NeuroQuantum,
        cut_layer: int,
        tokenizer: NeuroQuantumTokenizer,
        device: torch.device,
        lr: float = 3e-5,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.cut_layer = cut_layer

        # モデル分割
        self.client, self.server = split_model(model, cut_layer)
        self.client = self.client.to(device)
        self.server = self.server.to(device)

        # オプティマイザ（クライアントとサーバーそれぞれに）
        self.client_optimizer = torch.optim.AdamW(
            self.client.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.server_optimizer = torch.optim.AdamW(
            self.server.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.grad_clip = grad_clip

        logger.info(
            f"SplitLearningTrainer initialized: cut_layer={cut_layer}, "
            f"client_blocks={cut_layer}, server_blocks={model.config.num_layers - cut_layer}"
        )

    def train_step(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """
        1ステップの分割学習を実行する。

        フロー:
            1. クライアント: フォワードパス → 中間活性化を取得
            2. 中間活性化をサーバーに送信（ここではメモリ上で直接渡す）
            3. サーバー: フォワードパス → loss計算 → 逆伝播
            4. カットレイヤーの勾配をクライアントに返す
            5. クライアント: 勾配を使って逆伝播 → パラメータ更新
            6. サーバー: パラメータ更新

        Args:
            input_ids: (batch, seq) トークンID
            labels: (batch, seq) ラベル（-100はignore）

        Returns:
            loss値
        """
        self.client.train()
        self.server.train()

        # === Step 1: クライアント フォワードパス ===
        # requires_grad=Trueで中間活性化を保持（勾配を受け取るため）
        client_output = self.client(input_ids)

        # カットレイヤーの出力をdetachし、勾配を受け取れるようにする
        smashed_data = client_output.detach().requires_grad_(True)

        # === Step 2-3: サーバー フォワードパス + loss計算 ===
        logits = self.server(smashed_data)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.server.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # === Step 3: サーバー 逆伝播 ===
        self.server_optimizer.zero_grad()
        loss.backward()

        # === Step 4: カットレイヤーの勾配を取得 ===
        cut_grad = smashed_data.grad.clone()

        # === Step 5: クライアント 逆伝播 ===
        self.client_optimizer.zero_grad()
        client_output.backward(cut_grad)

        # === Step 6: 勾配クリッピング + パラメータ更新 ===
        torch.nn.utils.clip_grad_norm_(self.client.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.server.parameters(), self.grad_clip)
        self.client_optimizer.step()
        self.server_optimizer.step()

        return loss.item()

    def get_merged_model(self) -> NeuroQuantum:
        """学習済みの分割モデルを統合して完全なモデルに戻す。"""
        return merge_split_models(self.client, self.server, self.client.config)


# ========================================
# ネットワーク分散: サーバー
# ========================================

class SplitLearningServer:
    """
    分割学習サーバー。TCPソケットでクライアントからの
    中間活性化を受け取り、上位層の学習を行う。
    """

    def __init__(
        self,
        model: NeuroQuantum,
        cut_layer: int,
        device: torch.device,
        host: str = "0.0.0.0",
        port: int = 9000,
        lr: float = 3e-5,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
    ):
        self.device = device
        self.host = host
        self.port = port
        self.grad_clip = grad_clip

        # サーバー側モデル
        _, self.server_model = split_model(model, cut_layer)
        self.server_model = self.server_model.to(device)
        self.optimizer = torch.optim.AdamW(
            self.server_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.running = False
        self.total_steps = 0
        self.total_loss = 0.0

    def start(self):
        """サーバーを起動し、クライアントからの接続を待機する。"""
        self.running = True
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(1)
        logger.info(f"Split Learning Server listening on {self.host}:{self.port}")

        try:
            while self.running:
                server_sock.settimeout(1.0)
                try:
                    conn, addr = server_sock.accept()
                except socket.timeout:
                    continue
                logger.info(f"Client connected: {addr}")
                self._handle_client(conn)
                conn.close()
                logger.info(f"Client disconnected: {addr}")
        finally:
            server_sock.close()

    def _handle_client(self, conn: socket.socket):
        """クライアントとの通信ループ。"""
        while self.running:
            msg = recv_message(conn)
            if msg is None:
                break

            action = msg.get("action")

            if action == "train_step":
                result = self._process_train_step(msg)
                send_message(conn, result)

            elif action == "get_state":
                state = self.server_model.state_dict()
                # CPUに移動してから送信
                cpu_state = {k: v.cpu() for k, v in state.items()}
                send_message(conn, {"state_dict": cpu_state})

            elif action == "status":
                send_message(conn, {
                    "total_steps": self.total_steps,
                    "avg_loss": self.total_loss / max(self.total_steps, 1),
                })

            elif action == "shutdown":
                logger.info("Shutdown command received")
                self.running = False
                send_message(conn, {"status": "shutting_down"})
                break

            else:
                send_message(conn, {"error": f"Unknown action: {action}"})

    def _process_train_step(self, msg: Dict) -> Dict:
        """サーバー側の学習ステップを実行する。"""
        smashed_data = msg["smashed_data"].to(self.device).requires_grad_(True)
        labels = msg["labels"].to(self.device)

        self.server_model.train()

        # フォワードパス
        logits = self.server_model(smashed_data)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.server_model.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # 逆伝播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), self.grad_clip)
        self.optimizer.step()

        # カットレイヤーの勾配を返す
        cut_grad = smashed_data.grad.cpu()

        self.total_steps += 1
        self.total_loss += loss.item()

        return {
            "cut_grad": cut_grad,
            "loss": loss.item(),
            "step": self.total_steps,
        }


# ========================================
# ネットワーク分散: クライアント
# ========================================

class SplitLearningClient:
    """
    分割学習クライアント。下位層のフォワードパスを実行し、
    中間活性化をサーバーに送信する。
    """

    def __init__(
        self,
        model: NeuroQuantum,
        cut_layer: int,
        device: torch.device,
        server_host: str = "localhost",
        server_port: int = 9000,
        lr: float = 3e-5,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
    ):
        self.device = device
        self.server_host = server_host
        self.server_port = server_port
        self.grad_clip = grad_clip

        # クライアント側モデル
        self.client_model, _ = split_model(model, cut_layer)
        self.client_model = self.client_model.to(device)
        self.optimizer = torch.optim.AdamW(
            self.client_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.conn = None
        self.total_steps = 0

    def connect(self):
        """サーバーに接続する。"""
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.server_host, self.server_port))
        logger.info(f"Connected to server {self.server_host}:{self.server_port}")

    def disconnect(self):
        """サーバーとの接続を切断する。"""
        if self.conn:
            try:
                send_message(self.conn, {"action": "shutdown"})
                recv_message(self.conn)
            except Exception:
                pass
            self.conn.close()
            self.conn = None

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
        """
        1ステップの分割学習を実行する。

        Args:
            input_ids: (batch, seq) トークンID
            labels: (batch, seq) ラベル

        Returns:
            loss値
        """
        if self.conn is None:
            raise RuntimeError("Not connected to server. Call connect() first.")

        self.client_model.train()

        # クライアント側フォワードパス
        client_output = self.client_model(input_ids)

        # サーバーに送信（CPUに移動）
        send_message(self.conn, {
            "action": "train_step",
            "smashed_data": client_output.detach().cpu(),
            "labels": labels.cpu(),
        })

        # サーバーからの応答を受信
        response = recv_message(self.conn)
        if response is None:
            raise RuntimeError("Server disconnected")
        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")

        loss = response["loss"]
        cut_grad = response["cut_grad"].to(self.device)

        # クライアント側逆伝播
        self.optimizer.zero_grad()
        client_output.backward(cut_grad)
        torch.nn.utils.clip_grad_norm_(self.client_model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.total_steps += 1
        return loss

    def get_server_state(self) -> Dict:
        """サーバー側モデルのstate_dictを取得する。"""
        send_message(self.conn, {"action": "get_state"})
        response = recv_message(self.conn)
        return response.get("state_dict", {})

    def get_server_status(self) -> Dict:
        """サーバーの学習状況を取得する。"""
        send_message(self.conn, {"action": "status"})
        return recv_message(self.conn)
