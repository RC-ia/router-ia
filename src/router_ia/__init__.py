"""router-ia experimental GGUF MoE runtime."""

__version__ = "0.1.0"

# Load runtime optimizations before runners capture their original functions.
# This installs allocator hot-path fixes plus persistent KV/DeltaNet state.
from . import runtime_optimizations as _runtime_optimizations  # noqa: E402,F401

# The official stateful runner uses the same dedicated RoutedExpertCache path
# as the fused runner, without importing the fused module just to access it.
from . import qwen36_official_optimizations as _official_optimizations  # noqa: E402,F401

# Optional detailed profiler. It is completely inert unless QWEN36_PROFILE=1.
from . import qwen36_profiler as _profiler  # noqa: E402,F401
