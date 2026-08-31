# router-ia

Experimental inference runtime for GGUF Mixture-of-Experts models.

## Goal

Implement a minimal model executor that can keep MoE experts in a hierarchical memory cache instead of relying on static layer offloading:

```text
             Router
                |
        +-------+-------+
        |               |
      VRAM             RAM
   hot experts      warm experts
        |               |
        +-------+-------+
                |
             GGUF
```

The first target is `unsloth/Qwen3.6-35B-A3B-GGUF`.

## Target model

Qwen3.6-35B-A3B is a hybrid MoE model with:

- 40 transformer blocks
- 256 routed experts per MoE layer
- top-8 routed experts per token
- 1 shared expert per layer
- 2048 hidden size
- ~35B total parameters / ~3B active parameters per token
- 30 Gated DeltaNet layers + 10 full-attention layers

The runtime will initially target text-only inference and batch size 1.

## Initial milestones

1. Read GGUF metadata and enumerate tensors.
2. Detect and group expert tensors by layer/expert.
3. Implement an explicit VRAM/RAM expert cache with LRU eviction.
4. Implement the Qwen3.6 router and verify top-8 selections against a reference implementation.
5. Implement the minimum forward pass needed for one-token decode.
6. Add SSD as a cold tier only after VRAM/RAM works reliably.

## Constraints

- Do not modify `llama.cpp`.
- Keep the first implementation small and inspectable.
- Optimize for the user's hardware target: 4 GB VRAM + 8 GB RAM.
- Correctness first; performance comes later.
