# Inference Endpoint Crash: huggingface-hub / transformers Version Conflict

## Summary

The Hugging Face Inference Endpoint **tapiocaTakeshi/qubit-he-ro** fails to
start due to a package version conflict **inside** the pre-built container
image `inference-pytorch-cpu:sha-2d4f7f1`.

## Problem

The container image ships with **`huggingface-hub==1.6.0`**, but the bundled
version of **`transformers`** enforces a strict upper bound of
**`huggingface-hub<1.0`**. This causes an `ImportError` during Python startup
(the `transformers` dependency check), crashing the process before any model
code executes.

**Key facts:**

- This affects **all replicas** — the crash is 100% reproducible.
- Model files download successfully; the crash occurs seconds later at import
  time.
- This is a **container-level issue**, not a model or configuration problem.
- No endpoint-side setting change can work around this since the conflict is
  between two pre-installed packages.

## Affected Endpoint

- **Endpoint:** `tapiocaTakeshi/qubit-he-ro`
- **Container image:** `inference-pytorch-cpu:sha-2d4f7f1`

## Error

```
ImportError: transformers requires huggingface-hub<1.0,
             but huggingface-hub 1.6.0 is installed.
```

(Raised during `transformers` internal dependency validation at import time.)

## Workaround Applied

Pinned compatible versions in `requirements.txt` so that pip re-installs over
the container's broken packages:

```
huggingface_hub>=1.0.0,<2.0.0
transformers>=4.53.0
```

`transformers>=4.53.0` is compatible with `huggingface_hub>=1.0`.

## Action Requested (Hugging Face Support)

Please update the `inference-pytorch-cpu` container image to ship compatible
versions of `transformers` and `huggingface-hub`. The current combination
(`huggingface-hub==1.6.0` + a `transformers` version requiring `<1.0`) is
fundamentally broken.

Reference endpoint: **tapiocaTakeshi/qubit-he-ro**

To contact HF support, open a ticket at:
https://huggingface.co/support or email support@huggingface.co referencing
this endpoint and container image hash.
