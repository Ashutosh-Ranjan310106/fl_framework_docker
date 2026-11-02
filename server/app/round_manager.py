"""Round lifecycle state and stage timing manager for Flower Server."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
from typing import Dict, Optional
from shared.schemas.events import EventSeverity, EventType, RoundStageStatus

logger = logging.getLogger("server.round_manager")


class RoundManager:
    """Tracks round stage transitions, durations, and timestamps."""

    def __init__(self, experiment_id: str = "cifar10_baseline_001"):
        self.experiment_id = experiment_id
        self.current_round: int = 0
        self.total_rounds: int = 3
        self.round_start_times: Dict[int, float] = {}
        self.stage_start_times: Dict[str, float] = {}
        self.stage_statuses: Dict[int, Dict[str, str]] = {}

    def start_round(self, round_num: int) -> Dict[str, str]:
        """Mark round started and reset stage tracking."""
        self.current_round = round_num
        self.round_start_times[round_num] = time.time()
        self.stage_statuses[round_num] = {
            "model_distribution": RoundStageStatus.IN_PROGRESS.value,
            "client_training": RoundStageStatus.WAITING.value,
            "update_collection": RoundStageStatus.WAITING.value,
            "aggregation": RoundStageStatus.WAITING.value,
            "global_evaluation": RoundStageStatus.WAITING.value,
        }
        logger.info(f"========== Starting FL Round {round_num} ==========")
        return self.stage_statuses[round_num]

    def update_stage(self, stage_name: str, status: RoundStageStatus) -> float:
        """Update status of a round stage and calculate duration if completed."""
        round_num = self.current_round
        if round_num not in self.stage_statuses:
            self.stage_statuses[round_num] = {}

        self.stage_statuses[round_num][stage_name] = status.value
        now = time.time()

        if status == RoundStageStatus.IN_PROGRESS:
            self.stage_start_times[stage_name] = now
            return 0.0
        elif status == RoundStageStatus.COMPLETED and stage_name in self.stage_start_times:
            duration = now - self.stage_start_times[stage_name]
            return duration
        return 0.0

    def get_round_duration(self, round_num: int) -> float:
        """Return total elapsed time for given round."""
        start = self.round_start_times.get(round_num)
        return (time.time() - start) if start else 0.0
