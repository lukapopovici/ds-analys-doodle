"""Utility decorators that log to the shared Terminal logger for debugging.

Decorators included:
- log_call: logs function entry/exit, args and return value
- log_exceptions: logs exceptions and traceback
- timeit: measures execution time and logs it
- with_terminal: temporary enables terminal (optionally external) for the duration

These support both sync and async functions where applicable.
"""
from __future__ import annotations

import inspect
import time
import traceback
from functools import wraps
from typing import Any, Callable, Optional

from .terminal import terminal


def _format_args(args: tuple, kwargs: dict) -> str:
    parts = []
    if args:
        parts.extend(repr(a) for a in args)
    if kwargs:
        parts.extend(f"{k}={v!r}" for k, v in kwargs.items())
    return ", ".join(parts)


def log_call(level: str = "DEBUG", include_args: bool = True, include_return: bool = True) -> Callable:
    """Decorator: log calls to the function with optional args/return value.

    Args:
        level: Log level string (INFO, WARN, ERROR, DEBUG)
        include_args: Whether to include args/kwargs in the log
        include_return: Whether to include returned value in the log
    """

    def _decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def _async_wrapper(*args, **kwargs):
                if terminal.is_enabled() and include_args:
                    terminal.log(f"CALL {func.__name__}({_format_args(args, kwargs)})", level)
                elif terminal.is_enabled():
                    terminal.log(f"CALL {func.__name__}()", level)
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:  # let other decorators / handlers log this if needed
                    if terminal.is_enabled():
                        terminal.log(f"EXCEPTION in {func.__name__}: {exc}\n" + traceback.format_exc(), "ERROR")
                    raise
                if terminal.is_enabled() and include_return:
                    terminal.log(f"RETURN {func.__name__} -> {result!r}", level)
                return result

            return _async_wrapper

        @wraps(func)
        def _wrapper(*args, **kwargs):
            if terminal.is_enabled() and include_args:
                terminal.log(f"CALL {func.__name__}({_format_args(args, kwargs)})", level)
            elif terminal.is_enabled():
                terminal.log(f"CALL {func.__name__}()", level)

            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                if terminal.is_enabled():
                    terminal.log(f"EXCEPTION in {func.__name__}: {exc}\n" + traceback.format_exc(), "ERROR")
                raise

            if terminal.is_enabled() and include_return:
                terminal.log(f"RETURN {func.__name__} -> {result!r}", level)
            return result

        return _wrapper

    return _decorator


def log_exceptions(re_raise: bool = True) -> Callable:
    """Decorator: log any exception raised by the function with traceback.

    Args:
        re_raise: If True, re-raises the exception after logging (default).
    """

    def _decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def _async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if terminal.is_enabled():
                        terminal.log(f"EXCEPTION in {func.__name__}: {exc}\n" + traceback.format_exc(), "ERROR")
                    if re_raise:
                        raise
                    return None

            return _async_wrapper

        @wraps(func)
        def _wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if terminal.is_enabled():
                    terminal.log(f"EXCEPTION in {func.__name__}: {exc}\n" + traceback.format_exc(), "ERROR")
                if re_raise:
                    raise
                return None

        return _wrapper

    return _decorator


def timeit(level: str = "DEBUG", fmt: Optional[str] = None) -> Callable:
    """Decorator: measure and log function execution time.

    Args:
        level: Log level to use when writing the timing
        fmt: Optional format string with placeholders {name} and {elapsed}
    """

    def _decorator(func: Callable) -> Callable:
        message_fmt = fmt or "{name} took {elapsed:.6f}s"

        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def _async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - start
                    if terminal.is_enabled():
                        terminal.log(message_fmt.format(name=func.__name__, elapsed=elapsed), level)

            return _async_wrapper

        @wraps(func)
        def _wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if terminal.is_enabled():
                    terminal.log(message_fmt.format(name=func.__name__, elapsed=elapsed), level)

        return _wrapper

    return _decorator


def with_terminal(enabled: bool = True, use_external: bool = False) -> Callable:
    """Decorator: temporarily set Terminal enabled state for duration of function.

    This will restore previous Terminal state after function returns or raises.
    """

    def _decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def _async_wrapper(*args, **kwargs):
                prev_enabled = terminal.is_enabled()
                prev_external = getattr(terminal, "_use_external_term", False)

                # Apply requested settings
                if enabled and not prev_enabled:
                    terminal.set_enabled(True)
                if use_external != prev_external:
                    terminal.set_use_external_terminal(use_external)

                try:
                    return await func(*args, **kwargs)
                finally:
                    # Restore
                    terminal.set_use_external_terminal(prev_external)
                    terminal.set_enabled(prev_enabled)

            return _async_wrapper

        @wraps(func)
        def _wrapper(*args, **kwargs):
            prev_enabled = terminal.is_enabled()
            prev_external = getattr(terminal, "_use_external_term", False)

            if enabled and not prev_enabled:
                terminal.set_enabled(True)
            if use_external != prev_external:
                terminal.set_use_external_terminal(use_external)

            try:
                return func(*args, **kwargs)
            finally:
                terminal.set_use_external_terminal(prev_external)
                terminal.set_enabled(prev_enabled)

        return _wrapper

    return _decorator


__all__ = ["log_call", "log_exceptions", "timeit", "with_terminal"]
