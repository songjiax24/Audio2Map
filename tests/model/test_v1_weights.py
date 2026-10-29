"""A checkpoint from before the stem joined the encoder still resumes."""

from __future__ import annotations

import torch

from audio2map.model.v1.model import V1ChartModel
from audio2map.model.v1.weights import load_v1_optimizer, load_v1_state_dict
from audio2map.train.v1.loop import adamw_groups


def _legacy_weights(model: V1ChartModel) -> dict[str, torch.Tensor]:
    old: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        cloned = value.detach().clone()
        if key.startswith("encoder.stem."):
            old[key[len("encoder.") :]] = cloned
        elif key.startswith("encoder.layers."):
            old["encoder." + key[len("encoder.layers.") :]] = cloned
        else:
            old[key] = cloned
    return old


def _legacy_optimizer(model: V1ChartModel, opt: torch.optim.Optimizer) -> dict:
    names = [name for name, _ in model.named_parameters()]
    stem = [name for name in names if name.startswith("encoder.stem.")]
    layers = [name for name in names if name.startswith("encoder.layers.")]
    rest_skip = set(stem + layers + ["token_emb.weight"])
    rest = [name for name in names if name not in rest_skip]
    old_order = stem + ["token_emb.weight"] + layers + rest
    new_index = {name: index for index, name in enumerate(names)}
    current = opt.state_dict()
    groups = []
    for group in current["param_groups"]:
        copied = dict(group)
        # The saved file indexes parameters in the old order, 0..N-1.
        copied["params"] = list(range(len(old_order)))
        groups.append(copied)
    state = {}
    for old_index, name in enumerate(old_order):
        payload = current["state"].get(new_index[name])
        if payload is None:
            continue
        state[old_index] = {
            key: (value.detach().clone() if torch.is_tensor(value) else value)
            for key, value in payload.items()
        }
    return {"state": state, "param_groups": groups}


def test_legacy_checkpoint_restores_weights_and_adam_moments() -> None:
    torch.manual_seed(0)
    trained = V1ChartModel()
    opt = torch.optim.AdamW(trained.parameters(), lr=1e-3)
    for param in trained.parameters():
        param.grad = torch.randn_like(param)
    opt.step()
    fresh = V1ChartModel()
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    legacy = load_v1_state_dict(fresh, _legacy_weights(trained))
    assert legacy is True
    load_v1_optimizer(fresh, fresh_opt, _legacy_optimizer(trained, opt), legacy=True)
    trained_names = dict(trained.named_parameters())
    for name, param in fresh.named_parameters():
        assert torch.equal(param, trained_names[name])
        left = opt.state[trained_names[name]]["exp_avg"]
        right = fresh_opt.state[param]["exp_avg"]
        assert torch.equal(left, right)


def test_legacy_optimizer_keeps_matrix_decay_only() -> None:
    torch.manual_seed(0)
    trained = V1ChartModel()
    opt = torch.optim.AdamW(trained.parameters(), lr=1e-3, weight_decay=0.01)
    for param in trained.parameters():
        param.grad = torch.randn_like(param)
    opt.step()
    fresh = V1ChartModel()
    fresh_opt = torch.optim.AdamW(adamw_groups(fresh, 0.01), lr=3e-4)
    assert load_v1_state_dict(fresh, _legacy_weights(trained)) is True
    load_v1_optimizer(fresh, fresh_opt, _legacy_optimizer(trained, opt), legacy=True)
    trained_names = dict(trained.named_parameters())
    for name, param in fresh.named_parameters():
        src = trained_names[name]
        assert torch.equal(param, src)
        assert torch.equal(opt.state[src]["exp_avg"], fresh_opt.state[param]["exp_avg"])
    assert fresh_opt.param_groups[0]["weight_decay"] == 0.01
    assert fresh_opt.param_groups[1]["weight_decay"] == 0.0
    bias = fresh.encoder.layers[0].attn.q_proj.bias
    assert any(bias is param for param in fresh_opt.param_groups[1]["params"])


def test_current_checkpoint_loads_unchanged() -> None:
    torch.manual_seed(1)
    trained = V1ChartModel()
    opt = torch.optim.AdamW(trained.parameters(), lr=1e-3)
    for param in trained.parameters():
        param.grad = torch.ones_like(param)
    opt.step()
    fresh = V1ChartModel()
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    saved = {key: value.clone() for key, value in trained.state_dict().items()}
    legacy = load_v1_state_dict(fresh, saved)
    assert legacy is False
    load_v1_optimizer(fresh, fresh_opt, opt.state_dict(), legacy=False)
    pairs = zip(trained.named_parameters(), fresh.named_parameters(), strict=True)
    for (name, src), (_, dst) in pairs:
        assert name
        assert torch.equal(src, dst)
        assert torch.equal(opt.state[src]["exp_avg"], fresh_opt.state[dst]["exp_avg"])
