"""Metrics manager for Flower Server global metrics tracking."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("server.metrics_manager")


class MetricsManager:
    """Stores history of global evaluation and client aggregated metrics."""

    def __init__(self, experiment_id: str = "cifar10_baseline_001"):
        self.experiment_id = experiment_id
        self.global_history: List[Dict[str, Any]] = []
        self.client_summaries: Dict[int, Dict[str, Any]] = {}

    def record_global_metrics(
        self,
        round_num: int,
        loss: float,
        accuracy: float,
        samples: int,
        duration_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """Record global evaluation metrics for a round."""
        metric_record = {
            "round": round_num,
            "loss": loss,
            "accuracy": accuracy,
            "samples": samples,
            "duration_seconds": duration_seconds,
        }
        self.global_history.append(metric_record)
        logger.info(
            f"Global Round {round_num} Evaluation: Loss={loss:.4f}, Accuracy={accuracy:.4f}, Samples={samples}"
        )
        return metric_record

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Return most recent global metrics record."""
        return self.global_history[-1] if self.global_history else None

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Return complete historical list of global metrics."""
        return list(self.global_history)
