from __future__ import annotations

"""Qwen3.6 layer fidelity probe with packed/split MoE checkpoint support.

This wrapper fixes a Transformers/checkpoint layout mismatch found in the
original probe. Qwen3.5/3.6 checkpoints may store routed experts either as
packed 3-D tensors (gate_up_proj/down_proj) or as individual expert weights.
Transformers v5-style Qwen3.5 models expose the packed parameters, while some
converted checkpoints expose expert.{id}.gate_proj/up_proj/down_proj.

The existing probe is reused unchanged for the actual layer/reference/router
comparison. We only replace its layer materializer so both checkpoint layouts
work.
"""

import torch

from . import qwen36_layer_fidelity_probe as probe


def _load_split_expert_tensor(root, weight_map, key: str) -> torch.Tensor:
    return probe._load_checkpoint_tensor(root, weight_map, key)


def _materialize_layer(root, layer: torch.nn.Module, layer_idx: int, device: str):
    weight_map = probe._checkpoint_index(root)
    prefix = probe.base.layer_prefix(layer_idx)
    layer.to_empty(device=device)

    loaded = 0
    missing: list[str] = []
    destinations = {**dict(layer.named_parameters()), **dict(layer.named_buffers())}

    def load_one(checkpoint_key: str, destination: torch.Tensor) -> bool:
        try:
            tensor = probe._load_checkpoint_tensor(root, weight_map, checkpoint_key)
        except KeyError:
            return False
        if tuple(tensor.shape) != tuple(destination.shape):
            raise RuntimeError(
                f"Shape mismatch for {checkpoint_key}: checkpoint={tuple(tensor.shape)} "
                f"model={tuple(destination.shape)}"
            )
        if destination.dtype.is_floating_point:
            tensor = tensor.to(dtype=destination.dtype)
        destination.copy_(tensor.to(device=device))
        return True

    with torch.no_grad():
        for local_name, destination in destinations.items():
            checkpoint_key = prefix + local_name

            # Normal path: checkpoint has exactly the same packed parameter.
            if load_one(checkpoint_key, destination):
                loaded += 1
                continue

            # Qwen3.5/3.6 split-expert fallback. Transformers exposes:
            #   mlp.experts.gate_up_proj [E, 2I, H]
            #   mlp.experts.down_proj     [E, H, I]
            # while some checkpoints store:
            #   mlp.experts.{E}.gate_proj.weight
            #   mlp.experts.{E}.up_proj.weight
            #   mlp.experts.{E}.down_proj.weight
            if local_name in {
                "mlp.experts.gate_up_proj",
                "mlp.experts.down_proj",
            }:
                if destination.ndim != 3:
                    missing.append(checkpoint_key)
                    continue

                num_experts = int(destination.shape[0])
                found = True
                for expert_id in range(num_experts):
                    if local_name.endswith("gate_up_proj"):
                        gate_key = f"{prefix}mlp.experts.{expert_id}.gate_proj.weight"
                        up_key = f"{prefix}mlp.experts.{expert_id}.up_proj.weight"
                        try:
                            gate = _load_split_expert_tensor(root, weight_map, gate_key)
                            up = _load_split_expert_tensor(root, weight_map, up_key)
                        except KeyError:
                            found = False
                            break

                        if gate.ndim != 2 or up.ndim != 2:
                            raise RuntimeError(
                                f"Unexpected split expert shapes at layer {layer_idx}, expert {expert_id}: "
                                f"gate={tuple(gate.shape)} up={tuple(up.shape)}"
                            )
                        if gate.shape != up.shape:
                            raise RuntimeError(
                                f"Gate/up shape mismatch at layer {layer_idx}, expert {expert_id}: "
                                f"gate={tuple(gate.shape)} up={tuple(up.shape)}"
                            )
                        packed = torch.cat((gate, up), dim=0)
                        if tuple(packed.shape) != tuple(destination[expert_id].shape):
                            raise RuntimeError(
                                f"Packed gate/up shape mismatch at layer {layer_idx}, expert {expert_id}: "
                                f"packed={tuple(packed.shape)} model={tuple(destination[expert_id].shape)}"
                            )
                        packed = packed.to(dtype=destination.dtype, device=device)
                        destination[expert_id].copy_(packed)
                    else:
                        down_key = f"{prefix}mlp.experts.{expert_id}.down_proj.weight"
                        try:
                            down = _load_split_expert_tensor(root, weight_map, down_key)
                        except KeyError:
                            found = False
                            break
                        if tuple(down.shape) != tuple(destination[expert_id].shape):
                            raise RuntimeError(
                                f"Down shape mismatch at layer {layer_idx}, expert {expert_id}: "
                                f"checkpoint={tuple(down.shape)} model={tuple(destination[expert_id].shape)}"
                            )
                        destination[expert_id].copy_(down.to(dtype=destination.dtype, device=device))

                if found:
                    loaded += 1
                else:
                    missing.append(checkpoint_key)
                continue

            missing.append(checkpoint_key)

    if missing:
        raise RuntimeError(
            f"Layer {layer_idx} is missing {len(missing)} checkpoint tensors. "
            f"First missing: {missing[:5]}"
        )
    return loaded, len(destinations)


def main() -> None:
    probe._materialize_layer = _materialize_layer
    probe.main()


if __name__ == "__main__":
    main()
