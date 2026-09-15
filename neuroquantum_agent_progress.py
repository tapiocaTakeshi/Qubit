"""RunPod adapter kept separate from the model and controller."""
import json


def dispatch_job(handler, data, job):
    if handler._resolve_action(data) != "agent":
        return handler(data)

    events = []

    def publish(event):
        # Import lazily so local/Hugging Face inference does not need RunPod.
        from runpod.serverless.modules.rp_progress import progress_update
        events.append(event)
        # Polling can miss updates; each snapshot includes all events so far.
        progress_update(job, json.dumps({"agent_event": event, "agent_events": events}, ensure_ascii=False))

    return handler._handle_agent(data, on_event=publish)
