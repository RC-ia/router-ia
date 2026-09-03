"""router-ia experimental GGUF MoE runtime."""

__version__ = "0.1.0"

# Load conservative hot-loop optimizations before any runner captures its
# original functions. This does not change model math or routing decisions.
from . import runtime_optimizations as _runtime_optimizations  # noqa: E402,F401
