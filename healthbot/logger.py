"""
Logging configuration and node execution tracking.
Provides observability for LangGraph workflow execution.
"""

import logging
import time
from collections.abc import Callable
from functools import wraps

from healthbot.config import settings

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("healthbot")


def log_node_execution(node_name: str) -> Callable:
    """
    Decorator to log node execution with timing information.

    Args:
        node_name: Name of the LangGraph node being executed

    Returns:
        Decorated function with logging and timing
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state):
            start_time = time.time()
            logger.info(f"[START] Node: {node_name}")

            try:
                result = func(state)
                latency = time.time() - start_time

                logger.info(f"[END] Node: {node_name} | Latency: {latency:.2f}s")

                # Track latency in state if node_latencies exists
                if isinstance(result, dict) and "node_latencies" in result:
                    if result["node_latencies"] is None:
                        result["node_latencies"] = {}
                    result["node_latencies"][node_name] = latency

                return result

            except Exception as e:
                latency = time.time() - start_time
                logger.error(
                    f"[ERROR] Node: {node_name} | "
                    f"Latency: {latency:.2f}s | "
                    f"Error: {e!s}"
                )
                raise

        return wrapper

    return decorator
