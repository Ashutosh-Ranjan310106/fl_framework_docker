"""Unit tests for Server and Client Event Publishers."""

import time
import pytest
from server.app.event_publisher import EventPublisher
from client.app.event_publisher import ClientEventPublisher
from shared.schemas.events import EventSeverity, EventType


def test_server_event_publisher_enqueue():
    # Test with monitoring disabled so no real HTTP call is attempted
    pub = EventPublisher(enabled=False)
    pub.emit(
        event_type=EventType.ROUND_STARTED,
        round_num=1,
        severity=EventSeverity.INFO,
        message="Testing round started event",
    )
    pub.close()


def test_client_event_publisher_enqueue():
    pub = ClientEventPublisher(client_id="client_001", enabled=False)
    pub.emit(
        event_type=EventType.CLIENT_TRAINING_STARTED,
        round_num=1,
        severity=EventSeverity.INFO,
        message="Testing client training started event",
    )
    pub.close()
