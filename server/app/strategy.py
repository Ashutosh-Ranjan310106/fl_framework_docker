"""Observable and modular Flower FedAvg strategy with telemetry hooks and lifecycle management."""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from server.app.event_publisher import EventPublisher
from server.app.metrics_manager import MetricsManager
from server.app.model_manager import ModelManager
from server.app.report_generator import ReportGenerator
from server.app.round_manager import RoundManager
from shared.schemas.events import EventSeverity, EventType, RoundStageStatus

logger = logging.getLogger("server.strategy")


class MonitoredFedAvg(FedAvg):
    """
    FedAvg strategy instrumented with lifecycle events, client update validation,
    and checkpointing for real-time monitoring.
    """

    def __init__(
        self,
        event_publisher: EventPublisher,
        round_manager: RoundManager,
        model_manager: ModelManager,
        metrics_manager: MetricsManager,
        report_generator: Optional[ReportGenerator] = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        num_rounds: int = 3,
        **kwargs,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=ndarrays_to_parameters(model_manager.get_initial_parameters()),
            **kwargs,
        )
        self.publisher = event_publisher
        self.round_manager = round_manager
        self.model_manager = model_manager
        self.metrics_manager = metrics_manager
        self.report_generator = report_generator
        self.num_rounds = num_rounds
        self.latest_round_clients: Dict[int, List[Dict[str, Any]]] = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Lifecycle: Round starts, global model distribution begins."""
        self.round_manager.start_round(server_round)
        self.publisher.emit(
            event_type=EventType.ROUND_STARTED,
            round_num=server_round,
            severity=EventSeverity.INFO,
            message=f"Round {server_round}/{self.num_rounds} started.",
            data={"round": server_round, "total_rounds": self.num_rounds},
        )

        self.round_manager.update_stage("model_distribution", RoundStageStatus.IN_PROGRESS)
        self.publisher.emit(
            event_type=EventType.MODEL_DISTRIBUTION_STARTED,
            round_num=server_round,
            severity=EventSeverity.INFO,
            message=f"Distributing global model for round {server_round}.",
            data={"round": server_round},
        )

        # Standard client selection
        fit_configurations = super().configure_fit(server_round, parameters, client_manager)

        # Inject server_round and experiment_id into config for each client
        updated_configs = []
        for client, fit_ins in fit_configurations:
            config = dict(fit_ins.config)
            config["server_round"] = server_round
            config["experiment_id"] = self.round_manager.experiment_id
            updated_configs.append((client, FitIns(fit_ins.parameters, config)))

        dist_duration = self.round_manager.update_stage("model_distribution", RoundStageStatus.COMPLETED)
        self.round_manager.update_stage("client_training", RoundStageStatus.IN_PROGRESS)
        self.round_manager.update_stage("update_collection", RoundStageStatus.IN_PROGRESS)

        return updated_configs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Lifecycle: Collect updates, validate, aggregate weights, save checkpoint."""
        self.round_manager.update_stage("client_training", RoundStageStatus.COMPLETED)
        self.round_manager.update_stage("update_collection", RoundStageStatus.COMPLETED)

        num_results = len(results)
        num_failures = len(failures)
        logger.info(f"Round {server_round}: Received {num_results} updates, {num_failures} failures.")

        self.publisher.emit(
            event_type=EventType.CLIENT_UPDATES_STARTED,
            round_num=server_round,
            severity=EventSeverity.INFO,
            message=f"Received {num_results} client updates for round {server_round}.",
            data={"received_count": num_results, "failed_count": num_failures},
        )

        # Basic model validation on updates
        valid_results = []
        initial_params = self.model_manager.get_initial_parameters()
        expected_shapes = [p.shape for p in initial_params]

        for client_proxy, fit_res in results:
            client_id = getattr(client_proxy, "cid", "unknown_client")
            try:
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                received_shapes = [p.shape for p in ndarrays]
                if received_shapes != expected_shapes:
                    logger.warning(f"Shape mismatch from client {client_id}: {received_shapes} vs {expected_shapes}")
                    self.publisher.emit(
                        event_type=EventType.CLIENT_UPDATE_REJECTED,
                        round_num=server_round,
                        client_id=client_id,
                        severity=EventSeverity.WARNING,
                        message=f"Rejected update from {client_id}: Parameter shapes mismatch.",
                        data={"client_id": client_id, "error": "shape_mismatch"},
                    )
                    continue

                valid_results.append((client_proxy, fit_res))
                self.publisher.emit(
                    event_type=EventType.CLIENT_UPDATE_RECEIVED,
                    round_num=server_round,
                    client_id=client_id,
                    severity=EventSeverity.SUCCESS,
                    message=f"Valid update received from {client_id} ({fit_res.num_examples} samples).",
                    data={"client_id": client_id, "samples": fit_res.num_examples, "metrics": fit_res.metrics},
                )
            except Exception as e:
                logger.error(f"Error validating update from client {client_id}: {e}")
                self.publisher.emit(
                    event_type=EventType.CLIENT_UPDATE_REJECTED,
                    round_num=server_round,
                    client_id=client_id,
                    severity=EventSeverity.ERROR,
                    message=f"Update validation failed for {client_id}: {e}",
                    data={"client_id": client_id, "error": str(e)},
                )

        if not valid_results:
            logger.error(f"Round {server_round}: No valid client updates to aggregate.")
            self.publisher.emit(
                event_type=EventType.ERROR_OCCURRED,
                round_num=server_round,
                severity=EventSeverity.ERROR,
                message=f"Aggregation aborted in round {server_round}: zero valid updates.",
            )
            return None, {}

        # Aggregation stage
        self.round_manager.update_stage("aggregation", RoundStageStatus.IN_PROGRESS)
        self.publisher.emit(
            event_type=EventType.AGGREGATION_STARTED,
            round_num=server_round,
            severity=EventSeverity.INFO,
            message=f"Aggregating {len(valid_results)} client updates with FedAvg.",
            data={"participating_clients": len(valid_results)},
        )

        agg_start = time.time()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, valid_results, failures)
        agg_duration = time.time() - agg_start

        self.round_manager.update_stage("aggregation", RoundStageStatus.COMPLETED)
        self.publisher.emit(
            event_type=EventType.AGGREGATION_FINISHED,
            round_num=server_round,
            severity=EventSeverity.SUCCESS,
            message=f"Aggregation finished for round {server_round} in {agg_duration:.2f}s.",
            data={"duration_seconds": agg_duration, "participating_clients": len(valid_results)},
        )

        # Update server global model & save checkpoint
        if aggregated_parameters is not None:
            updated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            self.model_manager.set_parameters(updated_ndarrays)
            ckpt_path = self.model_manager.save_checkpoint(
                round_num=server_round,
                metrics=aggregated_metrics,
                experiment_id=self.round_manager.experiment_id,
            )
            if self.report_generator:
                self.report_generator.record_checkpoint(str(ckpt_path))

        # Collect client details for run report
        client_metrics_list = []
        for client_proxy, fit_res in valid_results:
            cid = getattr(client_proxy, "cid", "unknown_client")
            cid_actual = fit_res.metrics.get("client_id", cid) if fit_res.metrics else cid
            client_metrics_list.append({
                "client_id": str(cid_actual),
                "samples": fit_res.num_examples,
                "loss": float(fit_res.metrics.get("loss", 0.0)) if fit_res.metrics and "loss" in fit_res.metrics else None,
                "accuracy": float(fit_res.metrics.get("accuracy", 0.0)) if fit_res.metrics and "accuracy" in fit_res.metrics else None,
                "duration_seconds": float(fit_res.metrics.get("duration_seconds", 0.0)) if fit_res.metrics and "duration_seconds" in fit_res.metrics else None,
            })
        self.latest_round_clients[server_round] = client_metrics_list

        return aggregated_parameters, aggregated_metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Centralized evaluation on server-side test set."""
        if server_round == 0:
            # Round 0 is initial evaluation before training begins
            logger.info("Evaluating initial global model (Round 0)...")

        self.round_manager.update_stage("global_evaluation", RoundStageStatus.IN_PROGRESS)
        self.publisher.emit(
            event_type=EventType.EVALUATION_STARTED,
            round_num=server_round,
            severity=EventSeverity.INFO,
            message=f"Global evaluation started for round {server_round}.",
            data={"round": server_round},
        )

        eval_start = time.time()
        ndarrays = parameters_to_ndarrays(parameters)
        loss, accuracy, num_samples = self.model_manager.evaluate_global(ndarrays)
        eval_duration = time.time() - eval_start

        self.round_manager.update_stage("global_evaluation", RoundStageStatus.COMPLETED)
        self.publisher.emit(
            event_type=EventType.EVALUATION_FINISHED,
            round_num=server_round,
            severity=EventSeverity.SUCCESS,
            message=f"Global evaluation finished: Loss={loss:.4f}, Accuracy={accuracy:.4f}.",
            data={
                "loss": loss,
                "accuracy": accuracy,
                "samples": num_samples,
                "duration_seconds": eval_duration,
            },
        )

        # Record metrics in manager
        self.metrics_manager.record_global_metrics(
            round_num=server_round,
            loss=loss,
            accuracy=accuracy,
            samples=num_samples,
            duration_seconds=eval_duration,
        )

        # Record round metrics and generate volume reports
        if self.report_generator and server_round > 0:
            participating = len(self.latest_round_clients.get(server_round, []))
            self.report_generator.record_round(
                round_num=server_round,
                loss=loss,
                accuracy=accuracy,
                samples=num_samples,
                duration_seconds=eval_duration,
                participating_clients=participating,
                client_metrics=self.latest_round_clients.get(server_round, []),
            )
            is_final = server_round >= self.num_rounds
            self.report_generator.generate_all_reports(status="COMPLETED" if is_final else "IN PROGRESS")

        # If round > 0, emit round completed
        if server_round > 0:
            total_duration = self.round_manager.get_round_duration(server_round)
            self.publisher.emit(
                event_type=EventType.ROUND_COMPLETED,
                round_num=server_round,
                severity=EventSeverity.SUCCESS,
                message=f"Round {server_round}/{self.num_rounds} completed in {total_duration:.2f}s.",
                data={
                    "round": server_round,
                    "loss": loss,
                    "accuracy": accuracy,
                    "duration_seconds": total_duration,
                },
            )

            # Check if experiment completed
            if server_round >= self.num_rounds:
                self.publisher.emit(
                    event_type=EventType.EXPERIMENT_COMPLETED,
                    severity=EventSeverity.SUCCESS,
                    message=f"Experiment completed successfully after {self.num_rounds} rounds.",
                    data={
                        "total_rounds": self.num_rounds,
                        "final_loss": loss,
                        "final_accuracy": accuracy,
                    },
                )

        return float(loss), {"accuracy": float(accuracy), "samples": num_samples}
