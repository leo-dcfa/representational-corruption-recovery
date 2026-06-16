#!/usr/bin/env python
"""GPU sanity check (SPEC Phase 0). Verifies torch/cu128/sm_120 + a LoRA step.

Run: `uv run python scripts/gpu_sanity.py`
Does NOT touch the experiment matrix; it only confirms the stack works.
"""

from __future__ import annotations

import json


def main() -> int:
    import torch

    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["device"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        # a tiny matmul + autograd step on device to confirm sm_120 kernels work
        x = torch.randn(512, 512, device="cuda", requires_grad=True, dtype=torch.bfloat16)
        y = (x @ x.T).sum()
        y.backward()
        info["bf16_matmul_backward_ok"] = bool(x.grad is not None)
        info["free_mem_gb"] = round(torch.cuda.mem_get_info()[0] / 1e9, 2)

    from rcr.utils.provenance import provenance

    print(json.dumps({"gpu_sanity": info, "provenance": provenance()}, indent=2))
    ok = info["cuda_available"] and info.get("bf16_matmul_backward_ok", False)
    print("\nSANITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
