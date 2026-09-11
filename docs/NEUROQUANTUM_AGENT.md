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
- `documents`: at most 10 request-local strings, 4000 characters each (API only).
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
