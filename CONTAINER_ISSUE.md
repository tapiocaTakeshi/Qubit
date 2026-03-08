# Inference Endpoint Startup Failure (Container Dependency Conflict)

## Summary

The endpoint fails during startup because the selected container image
`inference-pytorch-cpu:sha-2d4f7f1` contains an internal dependency mismatch.

- `huggingface-hub` in the image: `1.6.0`
- bundled `transformers` in the same image expects: `huggingface-hub>=0.30.0,<1.0`

Python detects this mismatch at import time and raises an `ImportError` before
application startup completes.

## Observed Behavior

- All 4 startup attempts failed with the same error within seconds.
- This is not transient/retry-resolvable.
- Model download is successful every time (init container completes normally).
- Crash occurs during app boot while importing `sentence_transformers` →
  `transformers` dependency checks.

## Root Cause

The conflict is inside the prebuilt container image, not in model artifacts or
endpoint configuration.

## Required Fix

Redeploy using an updated image tag where `transformers` and
`huggingface-hub` are mutually compatible.

Recommended next steps:

1. Contact Hugging Face support with the image hash and endpoint logs.
2. Check Inference Endpoint docs for a newer `pytorch-cpu` runtime image tag.
3. Redeploy with the updated image.
