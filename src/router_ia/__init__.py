"""router-ia experimental GGUF MoE runtime."""

__version__ = "0.1.0"

# Load runtime optimizations before runners capture their original functions.
# This installs allocator hot-path fixes plus persistent KV/DeltaNet state.
from . import runtime_optimizations as _runtime_optimizations  # noqa: E402,F401
