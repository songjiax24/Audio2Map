"""Load a v1 checkpoint from before the stem moved into ``AudioEncoder``."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def load_v1_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> bool:
    """Load weights. Returns whether ``state`` used the old ``stem.*`` names.

    Old keys are ``stem.conv*`` and ``encoder.{i}.*``. Current keys are
    ``encoder.stem.conv*`` and ``encoder.layers.{i}.*``.
    """
    legacy = "stem.conv1.weight" in state
    model.load_state_dict(_remap_state(state) if legacy else state)
    return legacy


def load_v1_optimizer(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    state: dict[str, Any],
    *,
    legacy: bool,
) -> None:
    """Load AdamW state onto ``opt``.

    A legacy file stores one group in the old parameter order. Moments move
    onto the current names, then into ``opt``'s groups. Weight decay stays
    whatever ``opt`` already has, so bias and norm are not decayed.
    """
    if legacy:
        state = _pack_named_state(model, opt, _permute_legacy_optimizer(model, state))
    opt.load_state_dict(state)


def _remap_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {_remap_key(key): value for key, value in state.items()}


def _remap_key(key: str) -> str:
    if key.startswith("stem."):
        return "encoder.stem." + key[len("stem.") :]
    if key.startswith("encoder."):
        return "encoder.layers." + key[len("encoder.") :]
    return key


def _old_name(name: str) -> str:
    if name.startswith("encoder.stem."):
        return name[len("encoder.") :]
    if name.startswith("encoder.layers."):
        return "encoder." + name[len("encoder.layers.") :]
    return name


def _legacy_order(model: nn.Module) -> list[str]:
    """Old ``parameters()`` names: stem, token embedding, then the eight layers."""
    names = [name for name, _ in model.named_parameters()]
    stem = [name for name in names if name.startswith("encoder.stem.")]
    layers = [name for name in names if name.startswith("encoder.layers.")]
    embedding = [name for name in names if name == "token_emb.weight"]
    taken = set(stem + layers + embedding)
    rest = [name for name in names if name not in taken]
    if len(embedding) != 1 or not stem or not layers:
        raise RuntimeError("v1 model parameters are not stem, layers, and token_emb")
    return [_old_name(name) for name in stem + embedding + layers + rest]


def _permute_legacy_optimizer(model: nn.Module, state: dict[str, Any]) -> dict[str, Any]:
    """Reindex Adam moments onto the current parameter order.

    ``load_state_dict`` zips the saved id list with the live parameters by
    position. The saved list is the old order. The live list is the new one.
    """
    new_names = [name for name, _ in model.named_parameters()]
    old_names = _legacy_order(model)
    if len(old_names) != len(new_names):
        raise RuntimeError(f"legacy parameter count {len(old_names)} != {len(new_names)}")
    new_index = {name: index for index, name in enumerate(new_names)}
    mapping = [new_index[_remap_key(name)] for name in old_names]
    if sorted(mapping) != list(range(len(mapping))):
        raise RuntimeError("legacy optimizer remap is not a permutation")
    saved_ids = [index for group in state["param_groups"] for index in group["params"]]
    if len(saved_ids) != len(mapping):
        raise RuntimeError("legacy optimizer parameters do not match the v1 model")
    stored = state["state"]
    new_state = {}
    for old_pos, saved_id in enumerate(saved_ids):
        payload = stored.get(saved_id)
        if payload is None:
            payload = stored.get(int(saved_id))
        if payload is not None:
            new_state[mapping[old_pos]] = payload
    new_groups = []
    cursor = 0
    for group in state["param_groups"]:
        copied = dict(group)
        width = len(group["params"])
        copied["params"] = list(range(cursor, cursor + width))
        cursor += width
        new_groups.append(copied)
    if cursor != len(mapping):
        raise RuntimeError("legacy optimizer parameters do not match the v1 model")
    return {"state": new_state, "param_groups": new_groups}


def _pack_named_state(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Place named-order moments into ``opt``'s groups.

    ``state["state"]`` is keyed by ``named_parameters()`` index. The live
    groups may split matrices from bias and norm. Group settings, including
    weight decay, are copied from ``opt``.
    """
    names = [name for name, _ in model.named_parameters()]
    by_name: dict[str, Any] = {}
    for index, payload in state["state"].items():
        by_name[names[int(index)]] = payload
    name_of = {id(param): name for name, param in model.named_parameters()}
    opt_names: list[str] = []
    new_state: dict[int, Any] = {}
    new_groups: list[dict[str, Any]] = []
    flat = 0
    for group in opt.param_groups:
        ids: list[int] = []
        for param in group["params"]:
            name = name_of.get(id(param))
            if name is None:
                raise RuntimeError("optimizer parameter is not in the v1 model")
            payload = by_name.get(name)
            if payload is not None:
                new_state[flat] = payload
            ids.append(flat)
            opt_names.append(name)
            flat += 1
        copied = {key: value for key, value in group.items() if key != "params"}
        copied["params"] = ids
        new_groups.append(copied)
    if set(opt_names) != set(names):
        raise RuntimeError("optimizer parameters do not match the v1 model")
    return {"state": new_state, "param_groups": new_groups}
