"""asyncio helpers for structured concurrency and fire-and-forget work."""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

log = logging.getLogger(__name__)


async def run_together(*coroutines: Coroutine[Any, Any, None]) -> None:
    """Run coroutines concurrently; if one fails, cancel the others and re-raise its error.

    Unlike a bare :class:`asyncio.TaskGroup`, the original exception is raised
    rather than an :class:`ExceptionGroup`, so callers can use ordinary
    ``except`` clauses.
    """
    try:
        async with asyncio.TaskGroup() as group:
            for coroutine in coroutines:
                group.create_task(coroutine)
    except BaseExceptionGroup as failures:
        raise _first_failure(failures) from None


def _first_failure(group: BaseExceptionGroup[BaseException]) -> BaseException:
    for error in group.exceptions:
        if isinstance(error, BaseExceptionGroup):
            return _first_failure(error)
        if not isinstance(error, asyncio.CancelledError):
            return error
    return group.exceptions[0]


def detach(future: asyncio.Future[Any]) -> None:
    """Let a future we will never await finish quietly (no "exception was never retrieved")."""
    future.add_done_callback(lambda done: done.cancelled() or done.exception())


class BackgroundTasks:
    """Owns fire-and-forget tasks: keeps them referenced, logs failures, cancels on shutdown."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    def spawn(self, coroutine: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        return task

    async def cancel_all(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _finished(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            log.error("background task %s failed", task.get_name(), exc_info=error)
