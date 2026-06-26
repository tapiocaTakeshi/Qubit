#!/usr/bin/env python3
"""
Qubit Chat - a Textual-based TUI chatbot CLI.

A full-screen terminal chat interface for the Qubit AI engine. Messages are
rendered as chat bubbles, replies are generated off the UI thread so the
interface stays responsive, and a small set of slash commands control the
session.

Run it::

    python -m tui_chat.app
    python -m tui_chat.app --backend echo

Slash commands (type in the input box):

    /help            show available commands
    /clear           clear the conversation
    /backend <name>  switch backend (quantum, echo)
    /quit            exit

Keys:

    Enter      send message
    Ctrl+L     clear conversation
    Ctrl+C     quit
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import List

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static

from .backend import Message, available_backends, load_backend


WELCOME = (
    "[b]Qubit Chat[/b] へようこそ :sparkles:\n\n"
    "メッセージを入力して [b]Enter[/b] で送信します。\n"
    "コマンド: [b]/help[/b]  [b]/clear[/b]  [b]/backend[/b]  [b]/quit[/b]"
)

HELP_TEXT = (
    "[b]コマンド一覧[/b]\n"
    "  [b]/help[/b]            このヘルプを表示\n"
    "  [b]/clear[/b]           会話履歴をクリア\n"
    "  [b]/backend <name>[/b]  バックエンドを切替 ("
    + ", ".join(available_backends())
    + ")\n"
    "  [b]/quit[/b]            終了\n\n"
    "[dim]Ctrl+L: クリア / Ctrl+C: 終了[/dim]"
)


class ChatMessage(Static):
    """A single chat bubble. Role drives the CSS styling."""

    def __init__(self, role: str, content: str) -> None:
        self._role = role
        self._content = content
        super().__init__(self._format(), classes=f"msg msg-{role}")

    def _format(self) -> str:
        stamp = datetime.now().strftime("%H:%M")
        if self._role == "user":
            who = "[b]You[/b]"
        elif self._role == "assistant":
            who = "[b]Qubit[/b]"
        else:
            who = "[b]System[/b]"
        return f"{who} [dim]{stamp}[/dim]\n{self._content}"

    def update_content(self, content: str) -> None:
        self._content = content
        self.update(self._format())


class QubitChatApp(App):
    """The Textual chat application."""

    TITLE = "Qubit Chat"
    SUB_TITLE = "Quantum Prefrontal Chatbot"

    CSS = """
    Screen {
        background: $surface;
    }

    #chat-view {
        padding: 1 2;
        height: 1fr;
    }

    .msg {
        margin: 1 0;
        padding: 1 2;
        width: 100%;
    }

    .msg-user {
        background: $primary 25%;
        border: round $primary;
        margin-left: 12;
    }

    .msg-assistant {
        background: $panel;
        border: round $secondary;
        margin-right: 12;
    }

    .msg-system {
        background: $boost;
        border: round $accent;
        color: $text-muted;
    }

    .msg-thinking {
        color: $warning;
    }

    #input {
        dock: bottom;
        margin: 0 1 1 1;
        border: round $accent;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear", "Clear"),
    ]

    def __init__(self, backend_name: str | None = None) -> None:
        super().__init__()
        self._backend = load_backend(backend_name)
        self.SUB_TITLE = f"backend: {self._backend.name}"
        self._history: List[Message] = []

    # -- layout ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(id="chat-view")
        yield Input(placeholder="メッセージを入力... (/help でコマンド一覧)", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self._add_message("system", WELCOME)
        self.query_one("#input", Input).focus()

    # -- helpers --------------------------------------------------------

    def _add_message(self, role: str, content: str) -> ChatMessage:
        widget = ChatMessage(role, content)
        view = self.query_one("#chat-view", VerticalScroll)
        view.mount(widget)
        widget.scroll_visible()
        view.scroll_end(animate=False)
        return widget

    # -- input handling -------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        if text.startswith("/"):
            self._handle_command(text)
            return

        self._add_message("user", text)
        self._history.append(Message("user", text))

        thinking = self._add_message("assistant", "[i]…考え中[/i]")
        thinking.add_class("msg-thinking")
        self._generate_reply(text, thinking)

    def _handle_command(self, text: str) -> None:
        parts = text.split()
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        if cmd in ("/quit", "/exit", "/q"):
            self.exit()
        elif cmd == "/clear":
            self.action_clear()
        elif cmd == "/help":
            self._add_message("system", HELP_TEXT)
        elif cmd == "/backend":
            self._switch_backend(arg)
        else:
            self._add_message(
                "system", f"未知のコマンド: [b]{cmd}[/b]  ([b]/help[/b] で一覧)"
            )

    def _switch_backend(self, name: str | None) -> None:
        if not name:
            self._add_message(
                "system",
                "使い方: [b]/backend <name>[/b]  利用可能: "
                + ", ".join(available_backends()),
            )
            return
        try:
            self._backend = load_backend(name)
        except Exception as exc:
            self._add_message("system", f"[b]エラー:[/b] {exc}")
            return
        self.SUB_TITLE = f"backend: {self._backend.name}"
        self._add_message("system", f"バックエンドを [b]{self._backend.name}[/b] に切替えました。")

    # -- worker: reply generation --------------------------------------

    def _generate_reply(self, text: str, placeholder: ChatMessage) -> None:
        """Run the backend off the UI thread, then update the bubble."""

        history_snapshot = list(self._history)

        def work() -> str:
            return self._backend.reply(text, history_snapshot)

        worker = self.run_worker(work, thread=True, exclusive=False)

        async def finalize() -> None:
            try:
                reply = await worker.wait()
            except Exception as exc:  # pragma: no cover - backend errors
                reply = f"[b]エラー:[/b] {exc}"
            placeholder.remove_class("msg-thinking")
            placeholder.update_content(reply)
            self._history.append(Message("assistant", reply))
            self.query_one("#chat-view", VerticalScroll).scroll_end(animate=False)

        self.run_worker(finalize(), exclusive=False)

    # -- actions --------------------------------------------------------

    def action_clear(self) -> None:
        self._history.clear()
        view = self.query_one("#chat-view", VerticalScroll)
        view.remove_children()
        self._add_message("system", "会話履歴をクリアしました。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qubit TUI chatbot CLI")
    parser.add_argument(
        "--backend",
        choices=available_backends(),
        default=None,
        help="Chat backend to use (default: auto-select best available)",
    )
    args = parser.parse_args()

    app = QubitChatApp(backend_name=args.backend)
    app.run()


if __name__ == "__main__":
    main()
