"""
Observability module for AI agent tracing and monitoring.

Supports Langfuse integration with optional enable/disable via environment variables.
"""

import os
import warnings
from typing import Optional, List, Any
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

def _is_tracing_enabled() -> bool:
    """Check if tracing is enabled via environment variable."""
    enable_tracing = os.getenv("ENABLE_TRACING", "true").lower()
    return enable_tracing in ["true", "1", "yes", "on"]


def _validate_langfuse_config() -> bool:
    """Validate Langfuse configuration."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        return False

    # Check if keys are placeholder values
    if "your-" in public_key or "your-" in secret_key:
        return False

    return True


# ============================================================================
# Langfuse Handler Management
# ============================================================================

_langfuse_handler: Optional[Any] = None
_tracing_enabled: bool = False
_initialization_attempted: bool = False


def _initialize_langfuse() -> Optional[Any]:
    """
    Initialize Langfuse callback handler.

    Returns:
        CallbackHandler instance if successful, None otherwise
    """
    global _langfuse_handler, _tracing_enabled, _initialization_attempted

    if _initialization_attempted:
        return _langfuse_handler

    _initialization_attempted = True

    # Check if tracing is enabled
    if not _is_tracing_enabled():
        print("[Observability] Tracing disabled via ENABLE_TRACING=false")
        _tracing_enabled = False
        return None

    # Validate configuration
    if not _validate_langfuse_config():
        warnings.warn(
            "[Observability] Langfuse credentials not configured or invalid. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env file. "
            "Continuing without tracing...",
            UserWarning
        )
        _tracing_enabled = False
        return None

    # Import and initialize Langfuse
    try:
        from langfuse.langchain import CallbackHandler

        _langfuse_handler = CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        _tracing_enabled = True

        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        print(f"[Observability] Langfuse tracing enabled: {host}")

        return _langfuse_handler

    except ImportError:
        warnings.warn(
            "[Observability] Langfuse package not installed. "
            "Run: pip install langfuse\n"
            "Continuing without tracing...",
            UserWarning
        )
        _tracing_enabled = False
        return None

    except Exception as e:
        warnings.warn(
            f"[Observability] Failed to initialize Langfuse: {e}\n"
            "Continuing without tracing...",
            UserWarning
        )
        _tracing_enabled = False
        return None


# ============================================================================
# Public API
# ============================================================================

def get_callbacks() -> List[Any]:
    """
    Get list of callback handlers for LangChain operations.

    Returns:
        List containing Langfuse handler if enabled, empty list otherwise

    Example:
        response = agent.invoke(
            {"messages": [HumanMessage(query)]},
            config={"callbacks": get_callbacks()}
        )
    """
    handler = _initialize_langfuse()
    return [handler] if handler else []


def is_tracing_enabled() -> bool:
    """
    Check if observability tracing is currently enabled.

    Returns:
        True if tracing is active, False otherwise
    """
    _initialize_langfuse()  # Ensure initialization has been attempted
    return _tracing_enabled


def get_langfuse_handler() -> Optional[Any]:
    """
    Get the Langfuse callback handler instance.

    Returns:
        CallbackHandler instance if initialized, None otherwise

    Example:
        handler = get_langfuse_handler()
        if handler:
            print(f"Last trace URL: {handler.get_trace_url()}")
    """
    return _initialize_langfuse()
