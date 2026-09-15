# Bounded agent mode

`action: "agent"` runs a read-only tool loop with the current Qubit checkpoint.
Each decision chooses one tool; the next decision sees its result. A final
inference synthesizes the answer. This is orchestration, not additional training.
The checkpoint must be able to emit JSON tool decisions. Protocol 3 allows one
JSON repair attempt inside the decision budget before falling back with a warning.
Protocols 1 and 2 retain immediate fallback. A fallback never pretends a tool ran.

```json
{"input":{"action":"agent","prompt":"(12+3)*4を計算して",
 "parameters":{"max_steps":3,"history":[],"documents":[]}}}
```

- `max_steps`: protocols 1/2 use 1-4 (default 3); protocol 3 uses 1-10 (default 10).
  At most this many decisions (including repair/rethink) and one final inference.
- `max_seconds`: 1-240 (default 180), checked before/after model and tool calls.
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
The 1–4 decision limit for protocols 1/2, repeat detection and sanitized failures remain.
Agent inference disables prose repetition/deduplication controls that can corrupt
JSON keys/strings. Ordinary chat keeps its existing defaults.

## Action / Observation controller (protocol 3)

`AgentController` in `neuroquantum_agent.py` owns the request-local state, tool
allowlist, decisions, observations, completion checks and progress events.
`EndpointHandler._handle_agent` only supplies model inference. `run_agent` remains
the compatible Python entry point. No model weights or training jobs are changed.

Use this request after deploying the updated Qubit worker:

```json
{"input":{"action":"agent","prompt":"三重県の明日の天気を調べて傘が必要か判断して","parameters":{
  "protocol":3,"max_steps":10,"max_seconds":180,"tool_choice":"web_search","history":[]}}}
```

Web search requires `BRAVE_SEARCH_API_KEY`; unavailable named tools are rejected
before inference. The model still needs to choose a specific location/date when
needed. Search snippets are not a dedicated weather API or a factual guarantee.

Model decisions use either of these envelopes:

```json
{"status":"continue","action":"calculator","arguments":{"expression":"120*2"}}
```

```json
{"status":"complete","answer":"2個で240円です。"}
```

After a tool executes, its real success/error observation goes into the next
decision prompt. `complete` returns the answer directly without another inference.
`rethink` with empty arguments consumes a decision without executing a tool.
`clarify` asks the user and stops; no tool loop runs while waiting for their reply.
Legacy `{tool,input}` and `{name,arguments}` calls are accepted in protocol 3 too.
Unknown actions, extra fields and malformed arguments cannot execute. An optional
`thought` field is discarded; neither raw model decisions nor private reasoning
are published. Repeated tool+input stops the loop. Failed tool observations can
lead to a different action. `required`/named tools must succeed before a final
answer is accepted. A successful tool call is not proof of answer correctness.

Omitting `protocol` retains protocol 1 and its small limits for existing clients.
The response still uses `agent.version=1`; new clients should check
`agent.protocol=3` before allowing more than four steps. The controller is shared
by all three protocols and emits progress for all of them.

### Progress for a chat timeline

Both RunPod entry points (`runpod_handler.py` and `handler.py`) use the adapter in
`neuroquantum_agent_progress.py`. It calls the SDK's progress updater with JSON
text. While `/status/{id}` says `IN_PROGRESS`, its `output` can be a progress
string, not a final result object. Parse it as:

```json
{"agent_event":{"sequence":1,"type":"started","label":"依頼を受け付けました","max_steps":10,"available_tools":["calculator"]},
 "agent_events":[{"sequence":1,"type":"started","label":"依頼を受け付けました","max_steps":10,"available_tools":["calculator"]}]}
```

Every update contains a cumulative snapshot so a polling client can reconstruct
missed steps. Use increasing `sequence` to ignore stale/duplicate updates, match
`action` to `observation` by `call_id`, and treat the terminal job result as
authoritative (SDK progress delivery is asynchronous). On `COMPLETED`, read
`output.generated_text` and `output.agent.events`. Do not treat an intermediate
progress string as a completed answer. UI rendering belongs to Qubit-ai-web,
which is a separate repository and is not modified by this change.

| Event | Display |
| --- | --- |
| `started` | 依頼を受け付けました |
| `decision` | 次の処理を選択中 |
| `action` | Web検索中／計算中／文書検索中 |
| `observation` | 処理結果を受け取りました (`status` distinguishes failure) |
| `retry`, `rethink` | 処理の指定／取得済みの情報を再確認中 |
| `answer` | 回答を作成中 (separate synthesis only) |
| `clarification` | 追加情報が必要です |
| `failed`, `finished` | 失敗／終了 (`status` and `stop_reason` distinguish outcomes) |

The final trace includes `decision_count` and `inference_count`. A direct final
answer counts as one decision/inference. Budget exhaustion performs at most one
synthesis; timeout/cancellation launches no more inference. `run_agent` accepts
trusted `on_event` and `cancelled` Python callbacks, not callbacks from request
JSON. Cancellation/deadline checks are cooperative and cannot interrupt an
already-running synchronous GPU kernel or blocking network call. RunPod job
cancellation still uses `/cancel/{id}`; the adapter does not pretend to provide a
RunPod cancellation signal to the local callback. SDK background delivery errors
may only appear in worker logs; final events remain available in the result.

Available tools remain calculator, configured Web search and request-local
document search. Host filesystem access and arbitrary code execution need a
separate isolated execution service and are not exposed by this controller.

Tests: `python -m pytest -q tests/test_neuroquantum_agent.py tests/test_agent_protocol.py tests/test_agent_controller.py`.
These use scripted models and mocked progress delivery, not a live GPU endpoint.

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
