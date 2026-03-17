"""Event bus for publish/subscribe pattern."""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ..domain.enums import EventType
from ..domain.events import Event


EventHandler = Callable[[Event], None]


class EventBus:
    """Simple synchronous event bus for the backtesting engine."""

    def __init__(self):
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        handlers = self._handlers[event_type]
        if handler in handlers:
            self._handlers[event_type] = [h for h in handlers if h is not handler]

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribed handlers."""
        for handler in self._handlers.get(event.event_type, []):
            handler(event)

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
