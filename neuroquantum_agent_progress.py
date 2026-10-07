"""RunPod adapter kept separate from the model and controller."""
import json

# Streamed actions publish {"<action>_event": e, "<action>_events": [...]}.
STREAMED_ACTIONS = ("agent", "analyst")


def dispatch_job(handler, data, job):
    action = handler._resolve_action(data)
    if action not in STREAMED_ACTIONS:
        return handler(data)

    events = []

    def publish(event):
        # Import lazily so local/Hugging Face inference does not need RunPod.
        from runpod.serverless.modules.rp_progress import progress_update
        events.append(event)
        # Polling can miss updates; each snapshot includes all events so far.
        progress_update(job, json.dumps({f"{action}_event": event, f"{action}_events": events}, ensure_ascii=False))

    run = handler._handle_agent if action == "agent" else handler._handle_analyst
    return run(data, on_event=publish)
