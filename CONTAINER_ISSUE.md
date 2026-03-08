# Inference Endpoint Startup Failure (Container Dependency Conflict)

## Summary

The endpoint fails during startup because the selected container image
`inference-pytorch-cpu:sha-2d4f7f1` contains an internal dependency mismatch.

- `huggingface-hub` in the image: `0.21.0`
- bundled `transformers` in the same image expects: `huggingface-hub>=0.30.0`

Python detects this mismatch at import time and raises an `ImportError` before
application startup completes.

## Observed Behavior

- All replica startup attempts failed with the same error within seconds.
- This is not transient/retry-resolvable.
- Model download is successful every time (init container completes normally).
- Crash occurs during app boot while importing `transformers`, which runs a
  version check on `huggingface-hub` and raises `ImportError`.
- The issue is **not** caused by the model (`tapiocaTakeshi/Qubit`) or its
  configuration — the model downloads correctly each time.
- The conflict is entirely within the container image itself.

## Root Cause

The conflict is inside the prebuilt managed container image, not in model
artifacts or endpoint configuration. Image tag `inference-pytorch-cpu:sha-2d4f7f1`
ships `huggingface-hub==0.21.0` alongside a `transformers` version that requires
`huggingface-hub>=0.30.0`. This is a broken image that needs to be fixed on the
platform side.

## Workaround Applied

Added `huggingface-hub>=0.30.0,<1.0` to `requirements.txt` so that the
Inference Endpoints runtime will `pip install` the compatible version on
container startup, overriding the broken bundled version.

## Recommended Next Steps

1. **Immediate**: Redeploy the endpoint — the updated `requirements.txt` should
   resolve the import error by upgrading `huggingface-hub` at startup.
2. **Platform**: Submit a support request to Hugging Face flagging image tag
   `inference-pytorch-cpu:sha-2d4f7f1` as broken due to the `huggingface-hub`
   version mismatch.
3. **Long-term**: When a new CPU / `intel-spr` image tag becomes available,
   switch the endpoint container configuration to the updated tag and remove the
   `huggingface-hub` pin from `requirements.txt`.
