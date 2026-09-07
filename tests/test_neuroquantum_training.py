"""Runpod microbatch training must match the same logical full batches."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from train_hf_dataset import parse_args, resolve_bf16, train_epoch


class TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=11)
        self.embedding = nn.Embedding(11, 6)
        self.output = nn.Linear(6, 11)

    def forward(self, tokens):
        return self.output(self.embedding(tokens))


@pytest.mark.parametrize("accumulation", [3, 8])
def test_variable_length_microbatches_and_partial_tail_match_full_batch(accumulation):
    with torch.random.fork_rng():
        torch.manual_seed(47)
        original = TokenModel()
    sequences = [[1, 2, 3], [1, 3, 4, 5, 6], [1, 7], [1, 2, 8, 9], [1, 3, 10]]
    models, losses = [], []
    for batch_size, grad_steps in ((1, accumulation), (accumulation, 1)):
        model = copy.deepcopy(original)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        with patch("random.shuffle", lambda values: None):
            losses.append(train_epoch(
                model, sequences.copy(), SimpleNamespace(pad_id=0), optimizer,
                batch_size, 8, 0, torch.device("cpu"),
                gradient_accumulation_steps=grad_steps, use_bf16=True,
            ))
        models.append(model)
    assert losses[0] == pytest.approx(losses[1], abs=1e-6)
    for actual, expected in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_empty_prediction_batches_do_not_produce_nan_or_update_weights():
    model = TokenModel()
    before = copy.deepcopy(model.state_dict())
    loss = train_epoch(model, [[], [1]], SimpleNamespace(pad_id=0),
                       torch.optim.SGD(model.parameters(), lr=0.1), 1, 8, 0, "cpu")
    assert loss == 0
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_runpod_flags_can_disable_checkpointing_and_bf16():
    args = parse_args(["--dataset-id", "test/data", "--attention-window", "128",
                       "--no-gradient-checkpointing", "--no-use-bf16"])
    assert args.attention_window == 128
    assert not args.gradient_checkpointing
    assert not args.use_bf16


@pytest.mark.parametrize("supported", [False, True])
def test_bf16_selection_uses_gpu_capability_and_honors_opt_out(supported):
    with patch.object(torch.cuda, "is_bf16_supported", return_value=supported):
        assert resolve_bf16(True, "cuda") is supported
        assert not resolve_bf16(True, "cpu")
        assert not resolve_bf16(False, "cuda")
