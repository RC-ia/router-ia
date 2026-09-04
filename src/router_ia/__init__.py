"""router-ia experimental GGUF MoE runtime."""

__version__ = "0.1.0"

# Runtime patches are loaded by an explicit runner after the target module is
# fully imported. Importing them here makes ``python -m ...qwen36_chat_batch``
# load the runner once before runpy executes it, creating duplicate module state.
