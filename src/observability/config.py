"""
Langfuse SDK v3 Observability Configuration

Note: LlamaIndex instrumentation via OpenInference requires llama-index >= 0.11.0
For older versions, we skip LlamaIndex instrumentation but keep Langfuse tracing
via the @observe decorator which works for all Python code.
"""
import os
import urllib.request
import urllib.error

# Global flag to track Langfuse availability
LANGFUSE_AVAILABLE = False


def _check_langfuse_reachable(host: str) -> bool:
    """Check if Langfuse server is reachable."""
    try:
        # Try to reach the Langfuse health endpoint
        health_url = f"{host}/api/public/health"
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def setup_observability():
    """
    Configures Langfuse SDK v3 observability.
    
    Note: The @observe decorator in main.py provides tracing for all endpoints.
    LlamaIndex-specific instrumentation is optional and requires compatible versions.
    """
    global LANGFUSE_AVAILABLE
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    if not (public_key and secret_key):
        print("Langfuse credentials not found. Observability disabled.")
        LANGFUSE_AVAILABLE = False
        return None

    # Check if Langfuse is reachable before trying to connect
    if not _check_langfuse_reachable(host):
        print(f"Langfuse server not reachable at {host}. Observability disabled.")
        LANGFUSE_AVAILABLE = False
        # Set env var to disable OTEL tracing
        os.environ["OTEL_SDK_DISABLED"] = "true"
        return None

    print(f"Initializing Langfuse SDK v3 observability (host: {host})...")
    
    # Only import and use Langfuse if reachable
    from langfuse import get_client
    
    # Get the Langfuse client (automatically configured from env vars)
    langfuse = get_client()
    
    # Verify connection
    try:
        auth_result = langfuse.auth_check()
        print(f"Langfuse auth check: {auth_result}")
        LANGFUSE_AVAILABLE = True
    except Exception as e:
        print(f"Langfuse auth check failed: {e}")
        LANGFUSE_AVAILABLE = False
        os.environ["OTEL_SDK_DISABLED"] = "true"
        return None
    
    # Try to set up LlamaIndex instrumentation if compatible versions are installed
    # This is optional - the @observe decorator provides tracing without this
    try:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        LlamaIndexInstrumentor().instrument()
        print("LlamaIndex instrumentation enabled via OpenInference")
    except ImportError:
        print("Note: LlamaIndex instrumentation not available (openinference not installed)")
        print("  Langfuse tracing still works via @observe decorators")
    except Exception as e:
        # Catch version compatibility errors gracefully
        print(f"Note: LlamaIndex instrumentation skipped due to version incompatibility")
        print(f"  Details: {type(e).__name__}: {str(e)[:100]}")
        print("  Langfuse tracing still works via @observe decorators")
    
    return langfuse


def is_langfuse_available() -> bool:
    """Returns whether Langfuse is available and configured."""
    return LANGFUSE_AVAILABLE

