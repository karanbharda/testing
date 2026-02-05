"""
Retry utilities for handling external API failures
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Callable, Any, Optional
import random

logger = logging.getLogger(__name__)


class RetryError(Exception):
    """Exception raised when all retry attempts are exhausted"""
    pass


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions on failure

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Exponential backoff multiplier
        jitter: Add random jitter to delay to prevent thundering herd
        exceptions: Tuple of exceptions to retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {str(e)}"
                        )
                        break

                    # Calculate delay with exponential backoff and jitter
                    actual_delay = current_delay
                    if jitter:
                        actual_delay += random.uniform(0, current_delay * 0.1)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {actual_delay:.2f}s..."
                    )

                    await asyncio.sleep(actual_delay)
                    current_delay *= backoff_factor

            raise RetryError(f"Function {func.__name__} failed after {max_attempts} attempts") from last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {str(e)}"
                        )
                        break

                    # Calculate delay with exponential backoff and jitter
                    actual_delay = current_delay
                    if jitter:
                        actual_delay += random.uniform(0, current_delay * 0.1)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {actual_delay:.2f}s..."
                    )

                    time.sleep(actual_delay)
                    current_delay *= backoff_factor

            raise RetryError(f"Function {func.__name__} failed after {max_attempts} attempts") from last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exceptions: tuple = (Exception,)
):
    """
    Circuit breaker decorator to prevent cascading failures

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exceptions: Exceptions that count as failures
    """
    def decorator(func: Callable) -> Callable:
        # Circuit breaker state
        failures = 0
        last_failure_time = 0
        state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            nonlocal failures, last_failure_time, state

            current_time = time.time()

            if state == "OPEN":
                if current_time - last_failure_time > recovery_timeout:
                    state = "HALF_OPEN"
                    logger.info(f"Circuit breaker for {func.__name__} entering HALF_OPEN state")
                else:
                    raise RetryError(f"Circuit breaker OPEN for {func.__name__}")

            try:
                result = await func(*args, **kwargs)

                if state == "HALF_OPEN":
                    # Success in half-open state - close circuit
                    state = "CLOSED"
                    failures = 0
                    logger.info(f"Circuit breaker for {func.__name__} CLOSED after successful recovery")

                return result

            except expected_exceptions as e:
                failures += 1
                last_failure_time = current_time

                if failures >= failure_threshold:
                    state = "OPEN"
                    logger.error(f"Circuit breaker for {func.__name__} OPENED after {failures} failures")

                raise e

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            nonlocal failures, last_failure_time, state

            current_time = time.time()

            if state == "OPEN":
                if current_time - last_failure_time > recovery_timeout:
                    state = "HALF_OPEN"
                    logger.info(f"Circuit breaker for {func.__name__} entering HALF_OPEN state")
                else:
                    raise RetryError(f"Circuit breaker OPEN for {func.__name__}")

            try:
                result = func(*args, **kwargs)

                if state == "HALF_OPEN":
                    # Success in half-open state - close circuit
                    state = "CLOSED"
                    failures = 0
                    logger.info(f"Circuit breaker for {func.__name__} CLOSED after successful recovery")

                return result

            except expected_exceptions as e:
                failures += 1
                last_failure_time = current_time

                if failures >= failure_threshold:
                    state = "OPEN"
                    logger.error(f"Circuit breaker for {func.__name__} OPENED after {failures} failures")

                raise e

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Specific retry configurations for different types of operations
api_retry = retry_on_failure(
    max_attempts=3,
    delay=1.0,
    backoff_factor=2.0,
    exceptions=(ConnectionError, TimeoutError, OSError)
)

data_service_retry = retry_on_failure(
    max_attempts=5,
    delay=0.5,
    backoff_factor=1.5,
    exceptions=(ConnectionError, TimeoutError)
)

ml_service_retry = retry_on_failure(
    max_attempts=2,
    delay=2.0,
    backoff_factor=1.0,
    exceptions=(ConnectionError, TimeoutError)
)