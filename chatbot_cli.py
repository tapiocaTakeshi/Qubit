#!/usr/bin/env python3
"""
NeuroQuantum Chat CLI
Claude Code / Codex CLI 風のターミナルUI
"""

import sys
import os
import glob
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
# 優先順位: SFT（指示追従学習済み）> Pre-training（事前学習のみ）
CHECKPOINT_PREFIXES = ["megabyte_100mb_mathcode_sft", "neuroq_small_oasst_ja", "megabyte_100mb_sft", "megabyte_100mb_pretraining"]

# ========================================
# ANSI カラーコード
# ========================================
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

    # Foreground
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    # Bright
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"

def term_width():
    return shutil.get_terminal_size((80, 24)).columns

def clear_line():
    print("\r" + " " * term_width() + "\r", end="")

def print_banner():
    """起動バナー表示"""
    width = min(term_width(), 78)
    c = Color

    banner = f"""
{c.BRIGHT_CYAN}{c.BOLD}
   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗
   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗
   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║
   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║
   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝
   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝
{c.RESET}{c.DIM}   QUANTUM · nn.Embedding · QBNN{c.RESET}
"""
    print(banner)
    print(f"{c.GRAY}{'─' * width}{c.RESET}")
    print(f" {c.BOLD}NeuroQuantum Chat{c.RESET} {c.GRAY}v1.0.0{c.RESET}")
    print(f" {c.GRAY}日本語対応 AIチャットアシスタント{c.RESET}")
    print(f"{c.GRAY}{'─' * width}{c.RESET}\n")

def print_help():
    c = Color
    print(f"\n{c.BOLD} コマンド{c.RESET}")
    print(f"  {c.CYAN}/help{c.RESET}      このヘルプを表示")
    print(f"  {c.CYAN}/model{c.RESET}     使用モデルを表示")
    print(f"  {c.CYAN}/temp [値]{c.RESET}  生成のtemperatureを表示/設定 (例: /temp 0.7)")
    print(f"  {c.CYAN}/clear{c.RESET}     画面をクリア")
    print(f"  {c.CYAN}/stats{c.RESET}     会話統計を表示")
    print(f"  {c.CYAN}/exit{c.RESET}      終了 {c.GRAY}(quit, さようなら も可){c.RESET}\n")

def spinner_thinking(duration=0.6):
    """思考中スピナーアニメーション"""
    c = Color
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    start = time.time()
    i = 0
    while time.time() - start < duration:
        clear_line()
        print(f"{c.CYAN}{frames[i % len(frames)]}{c.RESET} {c.GRAY}考え中...{c.RESET}", end="", flush=True)
        time.sleep(0.05)
        i += 1
    clear_line()

def print_user_message(text):
    c = Color
    width = min(term_width(), 78)
    print(f"\n{c.BRIGHT_BLUE}{c.BOLD}▍ You{c.RESET}")
    print(f"{c.WHITE}{text}{c.RESET}")

def print_assistant_message(text, meta=None):
    c = Color
    print(f"\n{c.BRIGHT_MAGENTA}{c.BOLD}▍ NeuroQuantum{c.RESET}")
    for line in text.split("\n"):
        print(f"{c.WHITE}{line}{c.RESET}")
    if meta:
        print(f"{c.GRAY}  {meta}{c.RESET}")

def print_system_message(text, kind="info"):
    c = Color
    icons = {"info": ("ℹ", c.CYAN), "success": ("✓", c.GREEN), "error": ("✗", c.RED), "warn": ("⚠", c.YELLOW)}
    icon, color = icons.get(kind, ("•", c.GRAY))
    print(f"\n{color}{icon} {text}{c.RESET}")

def print_prompt():
    c = Color
    print()
    return input(f"{c.BRIGHT_GREEN}{c.BOLD}❯{c.RESET} ")

def find_latest_checkpoint():
    """
    CHECKPOINT_DIR から使用するチェックポイントを自動検出する。
    優先順位: SFT > Pre-training、各段階内では _best.pt > _merged.pt > _latest.pt > *_checkpoint.pt > 最新の *_epochN.pt
    """
    for prefix in CHECKPOINT_PREFIXES:
        candidates = [
            f"{CHECKPOINT_DIR}/{prefix}_best.pt",
            f"{CHECKPOINT_DIR}/{prefix}_merged.pt",
            f"{CHECKPOINT_DIR}/{prefix}_latest.pt",
            f"{CHECKPOINT_DIR}/{prefix}_checkpoint.pt",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        epoch_files = sorted(glob.glob(f"{CHECKPOINT_DIR}/{prefix}_epoch*.pt"))
        if epoch_files:
            return epoch_files[-1]

    return None

# ========================================
# チャットエンジン（embedding-Gemma使用）
# ========================================
class ChatEngine:
    def __init__(self):
        self.history = []
        self.turn_count = 0
        self.temperature = 0.8
        self.max_tokens = 60

        # NeuroQuantum (QBNN) 本体モデル（embedding-gemmaは内部の語彙初期化にのみ使用）
        self.qbnn_model = None
        self.qbnn_tokenizer = None
        self.qbnn_config = None
        self.qbnn_device = "cpu"
        self.qbnn_checkpoint_path = None

    def load_qbnn(self):
        """
        /workspace/checkpoints/ から学習済みNeuroQuantum(QBNN)モデルを読み込む。
        チェックポイントが見つからない場合はQBNN生成機能を無効化する。
        """
        import torch
        from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

        ckpt_path = find_latest_checkpoint()
        if ckpt_path is None:
            return False

        self.qbnn_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qbnn_tokenizer = NeuroQuantumTokenizer(vocab_size=32000, model_file="neuroq_tokenizer.model")
        self.qbnn_config = NeuroQuantumConfig(
            vocab_size=32000, embed_dim=1024, hidden_dim=2048,
            num_heads=16, num_layers=10, max_seq_len=512,
        )
        self.qbnn_model = NeuroQuantum(
            config=self.qbnn_config,
            tokenizer=self.qbnn_tokenizer,
        ).to(self.qbnn_device)

        state_dict = torch.load(ckpt_path, map_location=self.qbnn_device, weights_only=False)
        self.qbnn_model.load_state_dict(state_dict, strict=False)
        self.qbnn_model.eval()

        self.qbnn_checkpoint_path = ckpt_path
        return True

    def generate_qbnn(self, prompt: str, max_tokens: int = None, temperature: float = None,
                       repetition_penalty: float = 1.3, no_repeat_last_n: int = 32,
                       min_tokens: int = 6) -> str:
        """
        学習済みQBNNモデルでテキスト生成する（応答本体）。

        repetition_penalty: 既出トークンのロジットを弱める強さ（1.0で無効、大きいほど繰り返しを避ける）
        no_repeat_last_n: 直近何トークン分を繰り返し判定の対象にするか
        min_tokens: 新しいBOSの直後などでモデルが即座にEOSを出して空応答になるのを防ぐため、
                    最低限生成させるトークン数（この間はEOS/PAD/EOFを無視する）
        """
        import torch

        if self.qbnn_model is None:
            return ""

        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        input_ids = self.qbnn_tokenizer.encode(prompt, add_special=False)
        # ユーザー発話をBOS...EOSで一区切りとして閉じ、続けて新しいBOSから
        # 応答の生成を開始する（学習時のBOS〜EOS単位の文書構造に合わせる）
        generated = (
            [self.qbnn_tokenizer.bos_id] + input_ids + [self.qbnn_tokenizer.eos_id]
            + [self.qbnn_tokenizer.bos_id]
        )
        prompt_len = len(generated)
        stop_ids = (self.qbnn_tokenizer.eos_id, self.qbnn_tokenizer.pad_id, self.qbnn_tokenizer.eof_id)

        with torch.no_grad():
            for step in range(max_tokens):
                seq_len = len(generated)
                if seq_len >= self.qbnn_config.max_seq_len:
                    break
                padded = generated + [self.qbnn_tokenizer.pad_id] * (self.qbnn_config.max_seq_len - seq_len)
                input_tensor = torch.tensor([padded], dtype=torch.long, device=self.qbnn_device)
                logits = self.qbnn_model(input_tensor)
                next_logits = logits[0, seq_len - 1, :].clone()

                # 繰り返しペナルティ: 直近に出たトークンのロジットを弱める（CTRL方式）
                if repetition_penalty != 1.0:
                    recent_tokens = set(generated[-no_repeat_last_n:])
                    for tok_id in recent_tokens:
                        if next_logits[tok_id] > 0:
                            next_logits[tok_id] /= repetition_penalty
                        else:
                            next_logits[tok_id] *= repetition_penalty

                # 最低生成トークン数に達するまでは終端トークンを選ばせない（空応答防止）
                if step < min_tokens:
                    for tok_id in stop_ids:
                        next_logits[tok_id] = float('-inf')

                next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                if next_token in stop_ids:
                    break
                generated.append(next_token)

        # プロンプト部分を除いた生成分のみを応答として返す
        return self.qbnn_tokenizer.decode(generated[prompt_len:]).strip()

    def respond(self, user_input):
        """学習済みQBNNモデルの生成結果を応答本体として返す"""
        self.turn_count += 1
        self.history.append(("user", user_input))

        if self.qbnn_model is not None:
            response = self.generate_qbnn(user_input)
            if not response:
                response = "（うまく生成できませんでした。もう一度お試しください）"
        else:
            response = "QBNNモデルが読み込まれていないため生成できません。/model で状態を確認してください。"

        self.history.append(("assistant", response))
        return response

# ========================================
# メインループ
# ========================================
def main():
    c = Color
    os.system("clear" if os.name != "nt" else "cls")

    print_banner()

    engine = ChatEngine()

    print_system_message("NeuroQuantum(QBNN)チェックポイントを検索中...", "info")
    t0 = time.time()
    qbnn_loaded = engine.load_qbnn()
    elapsed = time.time() - t0
    if qbnn_loaded:
        print_system_message(
            f"QBNNモデル読み込み完了 ({elapsed:.1f}秒) : {engine.qbnn_checkpoint_path}", "success"
        )
    else:
        print_system_message(
            f"チェックポイントが見つかりません（{CHECKPOINT_DIR}）。QBNN生成は無効です。", "warn"
        )

    print(f"\n{c.GRAY}「/help」でコマンド一覧、「/exit」で終了{c.RESET}")

    while True:
        try:
            user_input = print_prompt().strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{c.GRAY}チャットを終了しました。{c.RESET}\n")
            break

        if not user_input:
            continue

        # コマンド処理
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ["/exit", "/quit"]:
                print_system_message("チャットを終了します。", "info")
                break
            elif cmd == "/help":
                print_help()
                continue
            elif cmd == "/clear":
                os.system("clear" if os.name != "nt" else "cls")
                print_banner()
                continue
            elif cmd == "/model":
                if engine.qbnn_config is not None:
                    print_system_message(f"Embedding: nn.Embedding ({engine.qbnn_config.embed_dim}次元)", "info")
                else:
                    print_system_message("Embedding: nn.Embedding（設定値なし）", "warn")
                if engine.qbnn_model is not None:
                    print_system_message(f"QBNN: {engine.qbnn_checkpoint_path}", "info")
                else:
                    print_system_message("QBNN: 未読み込み（チェックポイントなし）", "warn")
                continue
            elif cmd == "/stats":
                print_system_message(f"会話ターン数: {engine.turn_count}", "info")
                continue
            elif cmd.startswith("/temp"):
                parts = user_input.split()
                if len(parts) == 2:
                    try:
                        engine.temperature = max(0.1, min(2.0, float(parts[1])))
                        print_system_message(f"temperature を {engine.temperature:.2f} に設定しました", "success")
                    except ValueError:
                        print_system_message("数値を指定してください（例: /temp 0.9）", "error")
                else:
                    print_system_message(f"現在のtemperature: {engine.temperature:.2f}", "info")
                continue
            else:
                print_system_message(f"不明なコマンド: {user_input}", "error")
                continue

        if user_input.lower() in ["さようなら", "さよなら", "quit", "exit"]:
            print_assistant_message("またお話しましょう。さようなら！")
            break

        print_user_message(user_input)
        spinner_thinking(0.5)

        response = engine.respond(user_input)
        print_assistant_message(response, meta=f"⚛ QBNN生成 (temperature={engine.temperature:.2f})")

    c2 = Color
    print(f"{c2.GRAY}{'─' * min(term_width(), 78)}{c2.RESET}")
    print(f"{c2.GRAY}セッション終了 · {engine.turn_count} ターン{c2.RESET}\n")

if __name__ == "__main__":
    main()
