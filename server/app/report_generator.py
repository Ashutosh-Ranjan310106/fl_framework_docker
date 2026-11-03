"""Run report and metrics export generator for persistent volume storage."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

logger = logging.getLogger("server.report_generator")


class ReportGenerator:
    """
    Collects federated learning run statistics, accuracy progression,
    and client metrics, and exports reports (JSON, Markdown, CSV) to the volume.
    """

    def __init__(
        self,
        experiment_id: str,
        experiment_name: str = "CIFAR10 Flower Baseline",
        num_rounds: int = 3,
        strategy: str = "FedAvg",
        results_dir: str | Path = "results",
    ):
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.num_rounds = num_rounds
        self.strategy = strategy
        self.results_dir = Path(results_dir)
        self.reports_dir = self.results_dir / "run_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = datetime.now(timezone.utc).isoformat()
        self.end_time: Optional[str] = None
        self.rounds_data: Dict[int, Dict[str, Any]] = {}
        self.checkpoints: List[str] = []

    def record_round(
        self,
        round_num: int,
        loss: float,
        accuracy: float,
        samples: int,
        duration_seconds: float = 0.0,
        participating_clients: int = 0,
        client_metrics: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Record global evaluation and client metrics for a round."""
        record = {
            "round": round_num,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "samples": int(samples),
            "duration_seconds": float(duration_seconds),
            "participating_clients": int(participating_clients),
            "client_metrics": client_metrics or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.rounds_data[round_num] = record
        return record

    def record_checkpoint(self, checkpoint_path: str) -> None:
        """Register a model checkpoint file path."""
        if checkpoint_path not in self.checkpoints:
            self.checkpoints.append(str(checkpoint_path))

    def save_metrics_csv(self) -> Path:
        """Export round-by-round accuracy progression to CSV."""
        csv_path = self.reports_dir / f"{self.experiment_id}_metrics.csv"
        headers = ["round", "loss", "accuracy", "samples", "duration_seconds", "participating_clients", "timestamp"]

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in sorted(self.rounds_data.keys()):
                data = self.rounds_data[r]
                writer.writerow([
                    data["round"],
                    f"{data['loss']:.4f}",
                    f"{data['accuracy']:.4f}",
                    data["samples"],
                    f"{data['duration_seconds']:.2f}",
                    data["participating_clients"],
                    data["timestamp"],
                ])

        logger.info(f"Saved metrics CSV to {csv_path}")
        return csv_path

    def to_dict(self, status: str = "COMPLETED") -> Dict[str, Any]:
        """Generate structured dictionary representing the full run report."""
        now = datetime.now(timezone.utc).isoformat()
        if status in ("COMPLETED", "FAILED") and not self.end_time:
            self.end_time = now

        rounds_list = [self.rounds_data[r] for r in sorted(self.rounds_data.keys())]
        latest_round = rounds_list[-1] if rounds_list else None

        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "strategy": self.strategy,
            "status": status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_rounds_configured": self.num_rounds,
            "completed_rounds": len(rounds_list),
            "final_accuracy": latest_round["accuracy"] if latest_round else 0.0,
            "final_loss": latest_round["loss"] if latest_round else 0.0,
            "rounds": rounds_list,
            "checkpoints": self.checkpoints,
        }

    def save_json_report(self, status: str = "COMPLETED") -> Path:
        """Save detailed JSON report to volume."""
        report_data = self.to_dict(status=status)
        report_path = self.reports_dir / f"{self.experiment_id}_report.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # Also copy as latest_run_report.json
        latest_json = self.results_dir / "latest_run_report.json"
        try:
            shutil.copyfile(report_path, latest_json)
        except Exception:
            pass

        logger.info(f"Saved JSON run report to {report_path}")
        return report_path

    def save_markdown_report(self, status: str = "COMPLETED") -> Path:
        """Save formatted Markdown run report to volume."""
        report_data = self.to_dict(status=status)
        md_path = self.reports_dir / f"{self.experiment_id}_report.md"

        lines = [
            f"# Federated Learning Run Report: {self.experiment_name}",
            f"",
            f"**Experiment ID:** `{self.experiment_id}`  ",
            f"**Status:** `{status}`  ",
            f"**Strategy:** `{self.strategy}`  ",
            f"**Started:** {report_data['start_time']}  ",
            f"**Ended:** {report_data['end_time'] or 'In Progress'}  ",
            f"**Configured Rounds:** {self.num_rounds} | **Completed Rounds:** {report_data['completed_rounds']}  ",
            f"**Final Accuracy:** `{report_data['final_accuracy'] * 100:.2f}%`  ",
            f"**Final Loss:** `{report_data['final_loss']:.4f}`  ",
            f"",
            f"## 1. Global Accuracy & Loss Progression",
            f"",
            f"| Round | Global Loss | Global Accuracy (%) | Test Samples | Round Duration | Participating Clients |",
            f"| :---: | :---------: | :-----------------: | :----------: | :------------: | :-------------------: |",
        ]

        for r in report_data["rounds"]:
            acc_pct = r["accuracy"] * 100
            dur_str = f"{r['duration_seconds']:.2f}s"
            lines.append(
                f"| Round {r['round']} | {r['loss']:.4f} | {acc_pct:.2f}% | {r['samples']} | {dur_str} | {r['participating_clients']} |"
            )

        lines.extend([
            f"",
            f"## 2. Client Training Metrics by Round",
            f"",
        ])

        has_client_data = False
        for r in report_data["rounds"]:
            c_metrics = r.get("client_metrics", [])
            if c_metrics:
                has_client_data = True
                lines.append(f"### Round {r['round']} Clients")
                lines.append(f"| Client ID | Samples | Local Loss | Local Accuracy | Training Time |")
                lines.append(f"| :-------- | :-----: | :--------: | :------------: | :-----------: |")
                for cm in c_metrics:
                    cid = cm.get("client_id", "unknown")
                    sm = cm.get("samples", "--")
                    ls = f"{cm['loss']:.4f}" if cm.get("loss") is not None else "--"
                    ac = f"{cm['accuracy'] * 100:.2f}%" if cm.get("accuracy") is not None else "--"
                    tm = f"{cm['duration_seconds']:.2f}s" if cm.get("duration_seconds") is not None else "--"
                    lines.append(f"| `{cid}` | {sm} | {ls} | {ac} | {tm} |")
                lines.append("")

        if not has_client_data:
            lines.append("*No individual client breakdown recorded for this run.*\n")

        lines.extend([
            f"## 3. Saved Model Checkpoints",
            f"",
        ])
        if self.checkpoints:
            for ckpt in self.checkpoints:
                lines.append(f"- `{ckpt}`")
        else:
            lines.append("*No checkpoints recorded.*")

        lines.append("")

        content = "\n".join(lines)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Also copy as latest_run_report.md
        latest_md = self.results_dir / "latest_run_report.md"
        try:
            shutil.copyfile(md_path, latest_md)
        except Exception:
            pass

        logger.info(f"Saved Markdown run report to {md_path}")
        return md_path

    def generate_all_reports(self, status: str = "COMPLETED") -> Dict[str, str]:
        """Save CSV, JSON, and Markdown reports to the volume."""
        csv_file = self.save_metrics_csv()
        json_file = self.save_json_report(status=status)
        md_file = self.save_markdown_report(status=status)
        return {
            "csv": str(csv_file),
            "json": str(json_file),
            "markdown": str(md_file),
        }
