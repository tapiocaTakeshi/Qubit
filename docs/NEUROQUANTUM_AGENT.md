# Bounded agent mode

`action: "agent"` runs a read-only tool loop with the current Qubit checkpoint.
Each decision chooses one tool; the next decision sees its result. A final
inference synthesizes the answer. This is orchestration, not additional training.
The checkpoint must be able to emit JSON tool decisions. Invalid JSON falls back
to an ordinary answer with an explicit warning; it never pretends a tool ran.

```json
{"input":{"action":"agent","prompt":"(12+3)*4を計算して",
 "parameters":{"max_steps":3,"history":[],"documents":[]}}}
```

- `max_steps`: 1-4 (default 3); at most this many decisions and one final inference.
- `history`: at most 6 user/assistant messages, 2000 characters each.
- `documents`: at most 10 request-local strings, 4000 characters each. The web UI accepts one pasted text.
- Calculator: numbers, parentheses, `+ - * / %`; no eval, powers, imports or calls.
- Document search: existing BM25 index, no cross-request document persistence.
- Web search: optional `BRAVE_SEARCH_API_KEY` on the RunPod worker; fixed HTTPS
  Brave endpoint, 3 snippets per call, bounded response size, no redirects and
  no crawling of result URLs. Queries are sent to Brave and may incur API charges.

Output retains `generated_text` and adds `agent` (version 1): `status`, `steps`,
`warnings`, `sources`, `available_tools`. Steps describe executed tools, not
private reasoning. Sources are retrieved snippets, not proof of answer accuracy.
`completed` means the workflow finished, not that the task was factually verified.
Repeated tool calls stop the loop. Errors are shown without provider bodies or
credentials. No training, filesystem mutation, shell execution, or posting tools
are exposed. Tool output is labeled untrusted; prompt injection remains possible
at the answer layer, but cannot expand the tool allowlist.

Rebuild the repository Dockerfile and deploy the new image to RunPod first.
Both handler.py and runpod_handler.py accept parameters without extra mapping.
Use RunPod `/run` + `/status/{id}` for this multi-inference action. Cancel via
`/cancel/{id}`. Model context overflow rejects the request instead of silently
dropping the task or safety instructions.

Deploy the paired Qubit-ai-web change, then set `QUBIT_AGENT_ENABLED=true` on the
web server. This is deliberately opt-in to avoid old endpoints silently handling
the agent action as ordinary inference. Existing chat and training routes remain
unchanged. No live GPU training or deployment is performed by this code change.

Tests: `python -m pytest -q tests/test_neuroquantum_agent.py` (scripted inference,
no checkpoint or paid external calls). Before enabling publicly, evaluate JSON
tool selection with the deployed checkpoint and configure hosting access/rate
limits. The public web route inherits the deployment's existing access policy.

## Function-calling protocol 2

Set `parameters.protocol=2`. The response retains `agent.version=1` for the trace
envelope and acknowledges `agent.protocol=2`. Deploy the backend before the paired
web update: the web app rejects a worker that cannot acknowledge this protocol.

```json
{"input":{"action":"agent","prompt":"資料によると受付は何時？","parameters":{
  "protocol":2,"tool_choice":"document_search","max_steps":3,
  "documents":["受付は9時からです。"],"history":[]}}}
```

`tool_choice`: `auto`, `none`, `required`, `calculator`, `web_search`, or
`document_search`. Named choices restrict the allowlist. Required calls that do
not succeed cannot produce a speculative fallback. Missing search configuration
or documents rejects a named choice before inference. `none` skips tool decisions.

Function declarations use JSON Schema objects: exact keys, string length bounds,
and `additionalProperties: false`. Output can be
`{"name":"calculator","arguments":{"expression":"12*3"}}` or a single
OpenAI-style `tool_calls[].function` object (JSON-string arguments accepted).
Legacy `{tool,input}` is also accepted. Multiple calls, duplicate keys, unknown
functions and invalid arguments are rejected before execution. This is strict
validation, not grammar-constrained decoding or full OpenAI API compatibility.

`clarify(question)` returns a question and stops (`stop_reason=clarification`).
`finish({})` proceeds to answer synthesis. Trace entries have `call_id` and validated
`arguments`. Observations go to subsequent decisions and final answer synthesis.
The 1–4 decision limit, repeat detection, sanitized failures and cancellation remain.
Agent inference disables prose repetition/deduplication controls that can corrupt
JSON keys/strings. Ordinary chat keeps its existing defaults.

## Targeted SFT and a separate candidate

`train_agent.py` uses the runtime prompt builders, validates each record and masks
prompt tokens from loss: only assistant output and EOS/EOF are targets. Oversized
examples are rejected rather than truncating JSON or dropping instructions.
The synthetic curriculum has 239 records (166 train / 73 validation), grouped by
task so intermediate/final stages never leak between splits. Examples cover
arithmetic, document grounding, search failures, clarification, no-call conversation
and the reported off-topic follow-up. This is a small bootstrap curriculum, not
a broad dialogue corpus or evidence of improved model ability. Search failure
observations are simulated, not real provider logs.

Validate/export only (no Torch/GPU; output directory must be new):

```bash
python train_agent.py --output-dir /tmp/agent-curriculum
```

In a separate training process/pod, using the existing checkpoint and its matching
SentencePiece tokenizer, after stopping any competing checkpoint writer:

```bash
python train_agent.py --train --model-dir /runpod-volume \
  --output-dir /runpod-volume/agent-candidate-001 --epochs 1 --max-steps 20 --lr 0.00001
```

The loader uses only the explicit model directory, refuses vocabulary mismatches
and strictly loads weights (no partial resizing or random initialization).
`--dataset path.jsonl` optionally accepts the exported normalized contract, not
arbitrary HuggingFace rows. Review licenses and normalize function definitions,
arguments and observations before adding external corpora. Nothing is downloaded
automatically. Bounds: 10,000 records / 20 MB, 1–3 epochs, up to 200 optimizer
steps, batch size 1. Default is validation-only, not training.

Reports distinguish available records from actual `processed_samples`,
`target_tokens` and `optimizer_steps`. Training records held-out token loss before
and after, and writes `agent_candidate.pt`, never the serving checkpoint or volume
sync. Keep the source tokenizer with the candidate; its hash is in the report.
Before promotion, evaluate real tool selection, argument accuracy, clarification,
and multi-turn relevance on held-out tasks. Token loss is not an agent benchmark.
No model training or live endpoint evaluation was performed by this code change.
