#!/usr/bin/env python3
"""
Chat backends for the Qubit AI chatbot.

A backend is anything that turns a user message (plus the running
conversation) into an assistant reply string. Backends are intentionally
pluggable so the TUI works out of the box with zero heavy dependencies,
while still being able to drive the project's richer inference engines when
they are installed.

Resolution order (best available wins, override with ``--backend``):

1. ``neuroquantum``       - the project's pure-Python conversational engine
                            ``QuantumTextGenerator`` (``generative_ai.py``).
                            No external dependencies.
2. ``neuroquantum-model`` - the real torch-backed NeuroQuantum model from
                            ``inference.py`` (needs torch + a checkpoint).
3. ``quantum``            - the pure-Python judgment engine
                            ``QuantumFrontalLLM`` (``llm_inference.py``).
4. ``echo``               - a trivial fallback that always works.

Each backend exposes a single method::

    reply(message: str, history: list[Message]) -> str
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Protocol

# Make sure the project root is importable when this module is run from
# anywhere (e.g. ``python -m qubit_ai_cli.app`` or ``python qubit_ai_cli/app.py``).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@dataclass
class Message:
    """A single turn in the conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class Backend(Protocol):
    """The minimal interface every chat backend implements."""

    name: str

    def reply(self, message: str, history: List[Message]) -> str:
        """Return the assistant reply for ``message``."""
        ...


class EchoBackend:
    """Dependency-free fallback backend.

    Useful for testing the UI and as a guaranteed-to-load default.
    """

    name = "echo"
    description = "Simple echo backend (no dependencies)"

    def reply(self, message: str, history: List[Message]) -> str:
        turns = sum(1 for m in history if m.role == "user")
        return (
            f"あなたはこう言いました: 「{message}」\n\n"
            f"(echo backend / これは {turns} 回目のメッセージです)"
        )


class NeuroQuantumBackend:
    """Conversational backend powered by ``QuantumTextGenerator``.

    This wraps ``generative_ai.QuantumTextGenerator`` - the project's
    pure-Python NeuroQuantum conversational engine (intent detection,
    knowledge base, sentiment, quantum factor). It keeps its own internal
    conversation history, so multi-turn context works out of the box.
    """

    name = "neuroquantum"
    description = "NeuroQuantum conversational engine (pure Python)"

    def __init__(self) -> None:
        from generative_ai import QuantumTextGenerator

        self._gen = QuantumTextGenerator()

    def reply(self, message: str, history: List[Message]) -> str:
        return self._gen.generate(message)


class NeuroQuantumModelBackend:
    """Backend powered by the real torch-backed NeuroQuantum model.

    Loads the trained model and tokenizer via ``inference.load_model`` and
    generates with ``inference.generate``. Requires ``torch`` and a model
    checkpoint to be present, so it is opt-in (``--backend neuroquantum-model``)
    rather than part of auto-selection.
    """

    name = "neuroquantum-model"
    description = "NeuroQuantum trained model (torch, requires checkpoint)"

    def __init__(self) -> None:
        import inference

        self._inference = inference
        self._model, self._tokenizer, self._config, self._device = inference.load_model()

    def reply(self, message: str, history: List[Message]) -> str:
        return self._inference.generate(
            self._model,
            self._tokenizer,
            self._config,
            self._device,
            message,
        )


class QuantumBackend:
    """Backend powered by the project's pure-Python QuantumFrontalLLM.

    This reuses ``llm_inference.QuantumFrontalLLM`` so the TUI shares the
    same reasoning engine as the rest of Qubit without pulling in torch.
    """

    name = "quantum"
    description = "Qubit QuantumFrontalLLM engine (pure Python)"

    def __init__(self) -> None:
        from llm_inference import QuantumFrontalLLM

        self._llm = QuantumFrontalLLM()

    def reply(self, message: str, history: List[Message]) -> str:
        return self._llm.infer(message)


# Registry of backend factories, ordered from most to least preferred.
_BACKENDS = {
    "neuroquantum": NeuroQuantumBackend,
    "neuroquantum-model": NeuroQuantumModelBackend,
    "quantum": QuantumBackend,
    "echo": EchoBackend,
}

# Auto-selection order. The torch-backed model is intentionally excluded:
# it is heavy and needs a checkpoint, so it must be requested explicitly.
_PREFERENCE = ["neuroquantum", "quantum", "echo"]


def available_backends() -> List[str]:
    """Names of all registered backends."""
    return list(_BACKENDS.keys())


def load_backend(name: str | None = None) -> Backend:
    """Instantiate a backend by name, or auto-select the best available.

    Auto-selection walks ``_PREFERENCE`` and returns the first backend that
    imports and constructs cleanly, so a missing optional dependency simply
    degrades to the next option instead of crashing the app.
    """
    if name:
        if name not in _BACKENDS:
            raise ValueError(
                f"Unknown backend '{name}'. Choices: {', '.join(available_backends())}"
            )
        return _BACKENDS[name]()

    last_error: Exception | None = None
    for candidate in _PREFERENCE:
        try:
            return _BACKENDS[candidate]()
        except Exception as exc:  # pragma: no cover - depends on environment
            last_error = exc
            continue

    # _PREFERENCE always ends with "echo", which never fails, so we should
    # never get here. Guard anyway.
    raise RuntimeError(f"No chat backend could be loaded: {last_error}")
