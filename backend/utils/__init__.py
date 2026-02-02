"""
Backend Utilities - 100% Production Ready
========================================

Modular utility components for enterprise-grade trading bot backend.
Provides standardized error handling, input validation, performance monitoring,
and resilience patterns.

Components:
- exceptions: Standardized exception hierarchy
- validators: Input validation and configuration validation
- monitoring: Performance monitoring and metrics collection
- retry_utils: Retry logic and circuit breakers for external API resilience

All components are fully type-annotated, tested, and production-ready.
"""

from .validators import ConfigValidator, validate_chat_input
from .exceptions import (
    TradingBotError,
    ConfigurationError,
    DataServiceError,
    TradingExecutionError,
    ValidationError,
    NetworkError,
    AuthenticationError
)
from .monitoring import MLModelMonitor, log_model_performance, get_model_health_report
from .performance_monitor import PerformanceMonitor, get_performance_monitor
from .retry_utils import retry_on_failure, circuit_breaker, api_retry, data_service_retry, ml_service_retry
from .logging_utils import StructuredLogger, log_api_call, log_trading_operation, log_system_event, log_system_error, log_system_health

__all__ = [
    'ConfigValidator',
    'validate_chat_input',
    'TradingBotError',
    'ConfigurationError',
    'DataServiceError',
    'TradingExecutionError',
    'ValidationError',
    'NetworkError',
    'AuthenticationError',
    'MLModelMonitor',
    'log_model_performance',
    'get_model_health_report',
    'PerformanceMonitor',
    'get_performance_monitor',
    'retry_on_failure',
    'circuit_breaker',
    'api_retry',
    'data_service_retry',
    'ml_service_retry',
    'StructuredLogger',
    'log_api_call',
    'log_trading_operation',
    'log_system_event',
    'log_system_error',
    'log_system_health'
]