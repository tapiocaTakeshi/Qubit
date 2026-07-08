#!/usr/bin/env python3
"""
NeuroQuantum Chat CLI
Claude Code / Codex CLI 風のターミナルUI
"""

import sys
import os
import time
import shutil
import numpy as np

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
{c.RESET}{c.DIM}   QUANTUM · embedding-Gemma-300m · QBNN{c.RESET}
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

# ========================================
# チャットエンジン（embedding-Gemma使用）
# ========================================
class ChatEngine:
    def __init__(self):
        self.model = None
        self.history = []
        self.turn_count = 0
        self.knowledge = {
            "こんにちは": "こんにちは！何かお手伝いできることはありますか？",
            "あなたは誰": "私はNeuroQuantumです。QBNNとembedding-Gemma-300mを使用したAIアシスタントです。",
            "何ができる": "テキストの意味理解、質問応答、感情分析などができます。日本語に対応しています。",
            "機械学習とは": "機械学習は、データから自動的にパターンを学習し、予測や分類を行う技術です。",
            "量子とは": "量子は物理学における最小のエネルギー単位です。QBNNではこの概念を神経ネットワークに応用しています。",
            "日本について": "日本は東アジアに位置する島国で、豊かな伝統文化と最先端技術が共存しています。",
            "プログラミングとは": "プログラミングはコンピュータに指示を与えて問題を解決するスキルです。",
            "ありがとう": "どういたしまして！他に何かあればお気軽にどうぞ。",
        }

    def load_model(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("google/embeddinggemma-300m")

    def respond(self, user_input):
        from sklearn.metrics.pairwise import cosine_similarity

        self.turn_count += 1
        self.history.append(("user", user_input))

        user_emb = self.model.encode(user_input)
        keys = list(self.knowledge.keys())
        key_embs = self.model.encode(keys)

        sims = cosine_similarity([user_emb], key_embs)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score > 0.35:
            response = self.knowledge[keys[best_idx]]
        else:
            response = "興味深いですね。もう少し詳しく教えていただけますか？"

        self.history.append(("assistant", response))
        return response, best_score

# ========================================
# メインループ
# ========================================
def main():
    c = Color
    os.system("clear" if os.name != "nt" else "cls")

    print_banner()
    print_system_message("embedding-gemma-300m をロード中...", "info")

    engine = ChatEngine()
    t0 = time.time()
    engine.load_model()
    elapsed = time.time() - t0

    print_system_message(f"モデルロード完了 ({elapsed:.1f}秒)", "success")
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
                print_system_message("モデル: google/embeddinggemma-300m (300次元, 多言語対応)", "info")
                continue
            elif cmd == "/stats":
                print_system_message(f"会話ターン数: {engine.turn_count}", "info")
                continue
            else:
                print_system_message(f"不明なコマンド: {user_input}", "error")
                continue

        if user_input.lower() in ["さようなら", "さよなら", "quit", "exit"]:
            print_assistant_message("またお話しましょう。さようなら！")
            break

        print_user_message(user_input)
        spinner_thinking(0.5)

        response, score = engine.respond(user_input)
        print_assistant_message(response, meta=f"信頼度 {score:.1%}")

    c2 = Color
    print(f"{c2.GRAY}{'─' * min(term_width(), 78)}{c2.RESET}")
    print(f"{c2.GRAY}セッション終了 · {engine.turn_count} ターン{c2.RESET}\n")

if __name__ == "__main__":
    main()
