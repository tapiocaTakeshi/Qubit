import ast
from pathlib import Path


def test_abeja_dataset_is_explicitly_streaming_only():
    """Guard the handler contract without downloading the 588 GB dataset."""
    source = (Path(__file__).parents[1] / "handler.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert 'STREAMING_ONLY_DATASETS = {"kajuma/ABEJA-CC-JA"}' in source
    assert "if ds_id in STREAMING_ONLY_DATASETS:" in source
    assert 'load_kwargs["streaming"] = True' in source
