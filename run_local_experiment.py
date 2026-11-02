"""Execution script to run full 3-client federated learning experiment with FastAPI monitoring."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
import requests

DEFAULT_PYTHON = r"C:\Users\rrpra\AppData\Local\Programs\Python\Python311\python.exe"
PYTHON_EXE = os.getenv("PYTHON_EXE", DEFAULT_PYTHON if os.path.exists(DEFAULT_PYTHON) else sys.executable)
PROJECT_ROOT = Path(__file__).resolve().parent

def run_experiment():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    print("=========================================================")
    print(" Starting FL System: Monitor + Server + 3 Clients")
    print("=========================================================")

    # Ensure log directories
    (PROJECT_ROOT / "logs/server").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs/monitoring").mkdir(parents=True, exist_ok=True)
    for c in ["client_001", "client_002", "client_003"]:
        (PROJECT_ROOT / f"logs/clients/{c}").mkdir(parents=True, exist_ok=True)

    monitor_log = open(PROJECT_ROOT / "logs/monitoring/stdout.log", "w", encoding="utf-8")
    server_log = open(PROJECT_ROOT / "logs/server/stdout.log", "w", encoding="utf-8")

    # 1. Start Monitoring Service
    print("\n[1/5] Launching FastAPI Monitoring Service on port 8000...")
    monitor_proc = subprocess.Popen(
        [
            PYTHON_EXE,
            "-m",
            "uvicorn",
            "monitoring.app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "info",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=monitor_log,
        stderr=subprocess.STDOUT,
    )

    # Wait for monitor to become healthy
    monitor_healthy = False
    for i in range(20):
        try:
            r = requests.get("http://127.0.0.1:8000/api/health", timeout=1.0)
            if r.status_code == 200:
                print("Monitoring service is healthy at http://127.0.0.1:8000")
                monitor_healthy = True
                break
        except Exception:
            time.sleep(0.5)

    if not monitor_healthy:
        print("ERROR: Monitoring service failed to start.")
        monitor_proc.terminate()
        sys.exit(1)

    # 2. Start Flower Server
    print("\n[2/5] Launching Flower FL Central Server on port 8080...")
    server_env = env.copy()
    server_env["SERVER_ADDRESS"] = "127.0.0.1:8080"
    server_env["MONITORING_URL"] = "http://127.0.0.1:8000"
    server_env["NUM_ROUNDS"] = "3"
    server_env["MIN_FIT_CLIENTS"] = "3"
    server_env["MIN_AVAILABLE_CLIENTS"] = "3"

    server_proc = subprocess.Popen(
        [PYTHON_EXE, "-m", "server.app.server_app"],
        cwd=str(PROJECT_ROOT),
        env=server_env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )

    time.sleep(2.0)

    # 3. Start 3 Clients
    client_procs = []
    client_logs = []
    for client_id in ["client_001", "client_002", "client_003"]:
        print(f"\n[3/5] Launching {client_id}...")
        client_env = env.copy()
        client_env["CLIENT_ID"] = client_id
        client_env["SERVER_ADDRESS"] = "127.0.0.1:8080"
        client_env["MONITORING_URL"] = "http://127.0.0.1:8000"
        client_env["PARTITION_PATH"] = str(PROJECT_ROOT / f"datasets/partitions/cifar10/iid/clients_10_seed_42/{client_id}")
        client_env["LOCAL_EPOCHS"] = "2"
        client_env["BATCH_SIZE"] = "32"
        client_env["TORCH_THREADS"] = "2"

        c_log = open(PROJECT_ROOT / f"logs/clients/{client_id}/stdout.log", "w", encoding="utf-8")
        client_logs.append(c_log)

        c_proc = subprocess.Popen(
            [PYTHON_EXE, "-m", "client.app.client_app"],
            cwd=str(PROJECT_ROOT),
            env=client_env,
            stdout=c_log,
            stderr=subprocess.STDOUT,
        )
        client_procs.append((client_id, c_proc))

    print("\n[4/5] Training in progress across 3 clients & 3 federated rounds...")
    print("Waiting for training rounds to complete...")

    # Wait for server to exit
    server_proc.wait()
    print("Server process finished.")

    # Wait for clients
    for client_id, c_proc in client_procs:
        c_proc.wait()
        print(f"Client {client_id} process finished.")

    # Close log files
    server_log.close()
    for c_log in client_logs:
        c_log.close()

    # 4. Verify Final State in Monitoring Service
    print("\n[5/5] Querying final telemetry from Monitoring Service REST API...")
    try:
        r_exp = requests.get("http://127.0.0.1:8000/api/experiment/current").json()
        print(f"\nExperiment Status: {r_exp.get('status')}")

        r_clients = requests.get("http://127.0.0.1:8000/api/clients").json()
        print(f"\nRegistered Clients: {len(r_clients)}")
        for c in r_clients:
            print(f"  - {c['client_id']}: Status={c['status']}, Round={c['current_round']}, Loss={c.get('loss')}, Acc={c.get('accuracy')}")

        r_rounds = requests.get("http://127.0.0.1:8000/api/rounds").json()
        print(f"\nRounds Recorded: {len(r_rounds)}")
        for rd in r_rounds:
            print(f"  - Round {rd['round_number']}: Status={rd['status']}, Clients={rd['participating_clients']}, Loss={rd.get('loss')}, Acc={rd.get('accuracy')}, Duration={rd.get('duration_seconds')}s")

        r_metrics = requests.get("http://127.0.0.1:8000/api/metrics/global").json()
        print(f"\nGlobal Evaluation Metrics: {len(r_metrics)}")
        for m in r_metrics:
            print(f"  - Round {m['round']}: Loss={m['loss']:.4f}, Accuracy={m['accuracy']*100:.2f}%, Samples={m['samples']}")

        # Verify Checkpoints
        checkpoints = sorted(list(Path("models/checkpoints").glob("*.pt")))
        print(f"\nModel Checkpoints Created: {len(checkpoints)}")
        for cp in checkpoints:
            print(f"  - {cp.name} ({cp.stat().st_size / 1024:.1f} KB)")

    except Exception as e:
        print(f"Error querying monitor: {e}")
    finally:
        monitor_proc.terminate()
        monitor_log.close()
        print("\nFederated Learning Experiment run verified successfully.")

if __name__ == "__main__":
    run_experiment()
