"""Workflow utils."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import ParamSpec
from typing import TypeVar

T = TypeVar('T')
P = ParamSpec('P')


def retry_on_exception(
    wait_time: int,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry a function if an exception is raised.

    Parameters
    ----------
    wait_time: int
        Time to wait before retrying the function.
    """

    def decorator_retry(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper_retry(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                print(
                    f'Exception caught: {e}. \n'
                    f'Retrying after {wait_time} seconds...',
                )
                time.sleep(wait_time)
                return func(*args, **kwargs)

        return wrapper_retry

    return decorator_retry
