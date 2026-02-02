"""
Enhanced logging utilities for comprehensive system event tracking
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps
import traceback
import asyncio


class StructuredLogger:
    """Enhanced logger with structured logging capabilities"""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Add JSON formatter if not already present
        if not any(isinstance(h, logging.StreamHandler) and hasattr(h, 'formatter') for h in self.logger.handlers):
            handler = logging.StreamHandler()
            formatter = JSONFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_event(self, event_type: str, message: str, **kwargs):
        """Log a structured event"""
        extra = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.logger.info(message, extra=extra)

    def log_api_call(self, endpoint: str, method: str, status_code: int,
                    duration: float, user_id: Optional[str] = None, **kwargs):
        """Log API call events"""
        self.log_event(
            'api_call',
            f"API {method} {endpoint} - {status_code}",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration=duration,
            user_id=user_id,
            **kwargs
        )

    def log_trading_action(self, action: str, symbol: str, quantity: int,
                          price: float, order_type: str = "market", **kwargs):
        """Log trading actions"""
        self.log_event(
            'trading_action',
            f"Trading action: {action} {quantity} {symbol} at {price}",
            action=action,
            symbol=symbol,
            quantity=quantity,
            price=price,
            order_type=order_type,
            **kwargs
        )

    def log_error(self, error_type: str, message: str, traceback_info: Optional[str] = None, **kwargs):
        """Log error events"""
        if traceback_info is None:
            traceback_info = traceback.format_exc()

        self.log_event(
            'error',
            f"Error ({error_type}): {message}",
            error_type=error_type,
            traceback=traceback_info,
            **kwargs
        )

    def log_performance(self, operation: str, duration: float, success: bool = True, **kwargs):
        """Log performance metrics"""
        self.log_event(
            'performance',
            f"Performance: {operation} took {duration:.3f}s",
            operation=operation,
            duration=duration,
            success=success,
            **kwargs
        )

    def log_system_health(self, component: str, status: str, metrics: Optional[Dict[str, Any]] = None, **kwargs):
        """Log system health events"""
        self.log_event(
            'system_health',
            f"System health: {component} is {status}",
            component=component,
            status=status,
            metrics=metrics or {},
            **kwargs
        )


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        # Create base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }

        # Add any extra fields from the record
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno',
                             'pathname', 'filename', 'module', 'exc_info',
                             'exc_text', 'stack_info', 'lineno', 'funcName',
                             'created', 'msecs', 'relativeCreated', 'thread',
                             'threadName', 'processName', 'process', 'getMessage']:
                    log_entry[key] = value

        return json.dumps(log_entry, default=str)


def log_api_call(endpoint: str, method: str = "GET"):
    """Decorator to log API calls"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            logger = StructuredLogger(f"api.{func.__name__}")

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Try to extract status code from result if it's a response
                status_code = 200
                if hasattr(result, 'status_code'):
                    status_code = result.status_code

                logger.log_api_call(endpoint, method, status_code, duration)
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.log_api_call(endpoint, method, 500, duration, error=str(e))
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            logger = StructuredLogger(f"api.{func.__name__}")

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Try to extract status code from result if it's a response
                status_code = 200
                if hasattr(result, 'status_code'):
                    status_code = result.status_code

                logger.log_api_call(endpoint, method, status_code, duration)
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.log_api_call(endpoint, method, 500, duration, error=str(e))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def log_trading_operation(operation_type: str):
    """Decorator to log trading operations"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = StructuredLogger(f"trading.{func.__name__}")
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                logger.log_event(
                    'trading_operation',
                    f"Trading operation {operation_type} completed",
                    operation_type=operation_type,
                    duration=duration,
                    success=True
                )
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.log_error(
                    'trading_operation_failed',
                    f"Trading operation {operation_type} failed: {str(e)}",
                    operation_type=operation_type,
                    duration=duration
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = StructuredLogger(f"trading.{func.__name__}")
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                logger.log_event(
                    'trading_operation',
                    f"Trading operation {operation_type} completed",
                    operation_type=operation_type,
                    duration=duration,
                    success=True
                )
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.log_error(
                    'trading_operation_failed',
                    f"Trading operation {operation_type} failed: {str(e)}",
                    operation_type=operation_type,
                    duration=duration
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global logger instance
system_logger = StructuredLogger("trading_system")

# Convenience functions
def log_system_event(event_type: str, message: str, **kwargs):
    """Log a system event"""
    system_logger.log_event(event_type, message, **kwargs)

def log_system_error(error_type: str, message: str, **kwargs):
    """Log a system error"""
    system_logger.log_error(error_type, message, **kwargs)

def log_system_health(component: str, status: str, **kwargs):
    """Log system health"""
    system_logger.log_system_health(component, status, **kwargs)