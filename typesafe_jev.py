"""TypeSafe AI Jev integration for Qubit AI.

Jev is used as a typed decision layer, not as the text generator.
The endpoint contract is intentionally configurable because TypeSafe
deployments may expose different base URLs and response envelopes.

Environment variables:
  TYPESAFE_JEV_URL   Required API URL.
  TYPESAFE_API_KEY   Required bearer token.
  TYPESAFE_JEV_MODEL Optional model name, default: jev.
  TYPESAFE_JEV_TIMEOUT Optional timeout seconds, default: 20.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import requests


class TypeSafeJevError(RuntimeError):
    """Raised when the Jev request cannot be completed or decoded."""


class TypeSafeJevClient:
    """Small, dependency-light client for TypeSafe AI Jev."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.url = (url or os.getenv("TYPESAFE_JEV_URL", "")).strip()
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY", "")
        self.model = model or os.getenv("TYPESAFE_JEV_MODEL", "jev")
        self.timeout = float(timeout or os.getenv("TYPESAFE_JEV_TIMEOUT", "20"))

    @property
    def configured(self) -> bool:
        return bool(self.url and self.api_key)

    @staticmethod
    def _normalise_response(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            for key in ("result", "output", "judgment", "prediction", "data"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    merged = dict(nested)
                    merged.setdefault("raw", payload)
                    return merged
            return dict(payload)
        raise TypeSafeJevError("Jev returned a non-object JSON response")

    def judge(
        self,
        text: str,
        labels: Iterable[str],
        instruction: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a typed decision/probability object from Jev."""
        if not self.configured:
            raise TypeSafeJevError(
                "TypeSafe Jev is not configured. Set TYPESAFE_JEV_URL and "
                "TYPESAFE_API_KEY on the server."
            )
        labels_list = [str(label) for label in labels]
        if not labels_list:
            raise ValueError("labels must contain at least one category")

        body = {
            "model": self.model,
            "input": text,
            "text": text,
            "labels": labels_list,
            "instruction": instruction or "Classify the input using the supplied labels.",
        }
        if context:
            body["context"] = context

        try:
            response = requests.post(
                self.url,
                json=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = self._normalise_response(response.json())
        except requests.RequestException as exc:
            raise TypeSafeJevError(f"Jev request failed: {exc}") from exc
        except ValueError as exc:
            raise TypeSafeJevError("Jev returned invalid JSON") from exc

        result.setdefault("labels", labels_list)
        result.setdefault("model", self.model)
        return result


def judge_with_jev(
    text: str,
    labels: Iterable[str],
    instruction: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function used by the RunPod handler."""
    return TypeSafeJevClient().judge(text, labels, instruction, context)
