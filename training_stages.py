"""
Shared multi-stage 3B training schedule.

Used by both the `train_multistage_3b` handler action (handler.py, invoked
through RunPod serverless) and the local orchestrator script
(train_multistage_3b_runpod.py), so the two entry points can never drift
apart on what each stage actually trains on.

Stage order:
  1. FineWeb-2 Japanese
  2. ABEJA-CC-JA-edu
  3. Wikipedia (ja + en)
  4. Instruction
  5. Conversation
  6. Mathematics
  7. Code (tokyotech-llm/swallow-code-v2)

Each dataset entry's "id" is combined with its optional "split" as
"id:split" — this is the "owner/dataset:config" form that
EndpointHandler._load_custom_datasets already parses (the part after the
colon is passed to `datasets.load_dataset` as the `name`/config kwarg).
"""

TRAINING_STAGES = [
    {
        "name": "fineweb_japanese",
        "description": "FineWeb-2 Japanese",
        "mode": "general",
        "datasets": [
            {"id": "HuggingFaceFW/fineweb-2-edu-japanese", "max_samples": 1000000},
        ],
        "lr": 1e-4,
        "epochs": 1,
    },
    {
        "name": "abeja_cc_ja_edu",
        "description": "ABEJA-CC-JA-edu",
        "mode": "general",
        "datasets": [
            {"id": "ABEJA/abeja-cc-ja-edu", "max_samples": 500000},
        ],
        "lr": 8e-5,
        "epochs": 1,
    },
    {
        "name": "wikipedia",
        "description": "Wikipedia (Japanese + English)",
        "mode": "general",
        "datasets": [
            {"id": "wikimedia/wikipedia", "split": "20220301.ja", "max_samples": 200000},
            {"id": "wikimedia/wikipedia", "split": "20220301.en", "max_samples": 200000},
        ],
        "lr": 5e-5,
        "epochs": 1,
    },
    {
        "name": "instruction",
        "description": "Instruction-following datasets",
        "mode": "qa",
        "datasets": [
            {"id": "Open-Orca/OpenOrca", "max_samples": 100000},
            {"id": "HuggingFaceH4/ultrachat_200k", "max_samples": 100000},
        ],
        "lr": 3e-5,
        "epochs": 2,
    },
    {
        "name": "conversation",
        "description": "Conversational datasets",
        "mode": "qa",
        "datasets": [
            {"id": "kunishou/hh-rlhf-ja", "max_samples": 50000},
            {"id": "HuggingFaceH4/ultrachat_200k", "max_samples": 50000},
        ],
        "lr": 2e-5,
        "epochs": 2,
    },
    {
        "name": "mathematics",
        "description": "Mathematical reasoning datasets",
        "mode": "qa",
        "datasets": [
            {"id": "meta-math/MetaMathQA", "max_samples": 100000},
            {"id": "openai/gsm8k", "split": "main", "max_samples": 50000},
        ],
        "lr": 2e-5,
        "epochs": 2,
    },
    {
        "name": "code",
        "description": "Code datasets (tokyotech-llm/swallow-code-v2)",
        "mode": "general",
        "datasets": [
            {"id": "tokyotech-llm/swallow-code-v2", "max_samples": 5000000},
        ],
        "lr": 1e-5,
        "epochs": 1,
    },
]


def dataset_ids_for_stage(stage):
    """Build the "id:split" dataset_ids list _load_custom_datasets expects."""
    ids = []
    for ds in stage["datasets"]:
        spec = ds["id"]
        if ds.get("split"):
            spec = f"{spec}:{ds['split']}"
        ids.append(spec)
    return ids


def max_samples_for_stage(stage):
    """Per-dataset sample cap _load_custom_datasets applies uniformly per stage."""
    return max(ds.get("max_samples", 0) for ds in stage["datasets"])
