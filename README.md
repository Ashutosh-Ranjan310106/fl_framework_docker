# Docker-Based Flower Federated Learning Infrastructure with FastAPI Monitoring

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Flower FL](https://img.shields.io/badge/Flower-1.15.2-orange.svg)](https://flower.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)

An enterprise-ready, containerized, and observable **Federated Learning (FL)** framework built with **Flower**, **PyTorch**, **FastAPI**, **WebSockets**, and **Docker Compose**. The framework trains a deep convolutional neural network on distributed partitions of the **CIFAR-10** benchmark while streaming real-time client hardware metrics, training curves, and round lifecycle stages to an interactive dark-mode web console.

This framework is engineered as a foundational infrastructure for distributed machine learning research, edge simulation, and comprehensive **Federated Learning Security Testing & Benchmarking** (adversarial model poisoning, backdoor injection, Byzantine fault tolerance, and defense mechanism evaluation).

---

## 🌟 Framework Core Pillars & Key Capabilities

- **🧠 High-Efficiency PyTorch Engine**: Employs a custom, lightweight `CIFAR10CNN` model (2,193,674 trainable parameters) with Batch Normalization, Dropout regularization, and Cross-Entropy loss. Features bounded CPU thread allocation (`torch.set_num_threads(2)`) to eliminate multi-container host CPU thrashing.
- **🌐 Decoupled Microservice Architecture**: Clean three-tier separation between the central **Flower Server** (orchestration, validation, FedAvg aggregation, global test evaluation), independent **Federated Clients** (isolated data mounts, local optimization), and an asynchronous **FastAPI Monitoring Service** (SQLite persistence, REST API, and WebSocket streaming).
- **📊 Real-Time Observability & Live Web Console**: Sub-second full-duplex WebSocket telemetry powering an interactive dark-mode dashboard (Chart.js) with real-time global accuracy/loss progression, per-client loss comparisons, 5-stage round lifecycle trackers, and live client hardware profiling (CPU % and RAM MB via `psutil`).
- **⚡ Non-Blocking Telemetry Pipelines**: Dedicated background worker daemon threads with bounded queue buffers in both server and clients emit telemetry asynchronously via HTTP POST (`202 Accepted`), guaranteeing zero blocking or delay in PyTorch training loops or Flower gRPC communication.
- **🗂️ Advanced Data Heterogeneity & Partition Suite**: Integrated partition manager supporting deterministic **IID**, **Non-IID Dirichlet ($\alpha$)**, **Label Skew**, and **Quantity Skew** data distributions, backed by SHA-256 verification registries, statistics manifests, and an interactive **Streamlit Dataset Explorer**.
- **🐳 Dual-Mode Orchestration**: Run the full distributed ecosystem with a single command via **Docker Compose** (with automated health check barriers and bridge networking), or natively in the **Terminal** via an automated deadlock-free Python multi-process runner or individual shell sessions.
- **🛡️ Security & Benchmarking Foundation**: Built-in client role management (`honest`, with architectural hooks for malicious or Byzantine clients), server-side weight anomaly sanitization (detecting NaN/Inf parameters), and automated round-by-round PyTorch model checkpointing.

---

## 📑 Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Directory Structure](#2-project-directory-structure)
3. [Model Architecture & Federated Learning Dynamics](#3-model-architecture--federated-learning-dynamics)
4. [Prerequisites](#4-prerequisites)
5. [Windows 11 + WSL2 + Docker Desktop Setup](#5-windows-11--wsl2--docker-desktop-setup)
6. [Installation & Local Setup](#6-installation--local-setup)
7. [Docker Orchestration & Command Reference](#7-docker-orchestration--command-reference)
   - [7.1 Quick Commands Cheat Sheet](#71-quick-commands-cheat-sheet)
   - [7.2 Running the Unified Server & Monitoring Console](#72-running-the-unified-server--monitoring-console)
   - [7.3 Running Different Numbers of Clients (2, 3, 5, 10 Clients)](#73-running-different-numbers-of-clients-2-3-5-10-clients)
   - [7.4 Running External Clients (LAN, Wi-Fi, or Global IP)](#74-running-external-clients-lan-wi-fi-or-global-ip)
   - [7.5 Container Inspection, Logs & Top Resources](#75-container-inspection-logs--top-resources)
   - [7.6 Stopping, Cleaning Up & Recreating Containers](#76-stopping-cleaning-up--recreating-containers)
8. [Terminal & CLI Orchestration Reference](#8-terminal--cli-orchestration-reference)
9. [Live Monitoring Dashboard](#9-live-monitoring-dashboard)
10. [REST API Documentation](#10-rest-api-documentation)
11. [Inspecting Individual Clients](#11-inspecting-individual-clients)
12. [Viewing Round Lifecycle & History](#12-viewing-round-lifecycle--history)
13. [Stopping & Restarting Containers](#13-stopping--restarting-containers)
14. [Scaling to Additional Clients & Participation Settings](#14-scaling-to-additional-clients--participation-settings)
15. [Dataset Partitions & Management (Changing Datasets)](#15-dataset-partitions--management-changing-datasets)
   - [15.1 Switching Active Dataset Partitions via .env](#151-switching-active-dataset-partitions-via-env)
   - [15.2 Generating Heterogeneous Partitions via CLI](#152-generating-heterogeneous-partitions-via-cli)
   - [15.3 Interactive Dataset Explorer (Streamlit)](#153-interactive-dataset-explorer-streamlit)
   - [15.4 Adding a Custom Dataset (Format Specification)](#154-adding-a-custom-dataset-format-specification)
16. [Logs & Model Checkpoints](#16-logs--model-checkpoints)
17. [Troubleshooting Common Errors](#17-troubleshooting-common-errors)

---

## 1. Architecture Overview

The system consists of three decoupled application tiers orchestrating federated learning and observability:

```text
                 HOST MACHINE (Browser: http://localhost:8000)
                                      |
                           DOCKER COMPOSE / RUNTIME
                                      |
       +------------------------------+------------------------------+
       |                                                             |
FLOWER SERVER                                                 CLIENT NETWORK
       |                                                             |
   SuperLink (0.0.0.0:8080)                                      SuperNodes
       |                                                             |
   ServerApp                                                     ClientApp
  (FedAvg Strategy)                                      +-----------+-----------+
       |                                                 |           |           |
       |                                             Client 001  Client 002  Client 003
       |                                             (iid-part1) (iid-part2) (iid-part3)
       |                                                 |           |           |
       +------------------------------+------------------+-----------+-----------+
                                      |
                      Telemetry Events (HTTP POST 202)
                                      |
                                      v
                          FASTAPI MONITORING SERVICE
                                      |
                      +---------------+---------------+
                      |                               |
               SQLite Database                WebSocket Server
             (Events, Metrics,                 (/ws/live stream)
             Rounds, Clients)                         |
                                                      v
                                            Live Web Dashboard
                                           (Chart.js, Telemetry)
```

### Core Architecture Components:
- **Flower Server Application**: Orchestrates federated training rounds, validates client model updates, aggregates weights using FedAvg, executes centralized global evaluation against the isolated test dataset, and saves model checkpoints (`.pt`).
- **Federated Client Applications**: Independent client participants loading only their assigned CIFAR-10 partition mounted read-only (`:ro`). Clients execute local PyTorch training on CPU, measure loss, accuracy, duration, and hardware footprint (CPU & RAM), and return weight updates.
- **FastAPI Monitoring Service**: Independent asynchronous service exposing REST endpoints and WebSocket live streaming. Consumes events non-blockingly, persists state in SQLite, and pushes real-time updates to connected browsers.
- **Non-Blocking Background Event Publishers**: Both server and clients utilize background worker threads with bounded queues to emit telemetry to the monitor without blocking or corrupting model training loops.

---

## 2. Project Directory Structure

```text
fl_framework_docker/
│
├── server/
│   ├── app/
│   │   ├── server_app.py            # Central Flower ServerApp & gRPC server entry point
│   │   ├── strategy.py              # MonitoredFedAvg with lifecycle callbacks and update validation
│   │   ├── round_manager.py         # Round state transitions, durations, and stage tracker
│   │   ├── model_manager.py         # PyTorch checkpointing and server-side global evaluation
│   │   ├── metrics_manager.py       # Round evaluation and client aggregation history
│   │   └── event_publisher.py       # Asynchronous non-blocking event worker thread
│   ├── Dockerfile                   # Python 3.11-slim server image
│   └── requirements.txt             # Server dependencies (flwr, torch, etc.)
│
├── client/
│   ├── app/
│   │   ├── client_app.py            # Flower ClientApp & NumPyClient with reconnect logic
│   │   ├── trainer.py               # Local PyTorch training & evaluation loops (CPU)
│   │   ├── model.py                 # Standard CIFAR-10 CNN architecture (32x32 RGB -> 10 classes)
│   │   ├── dataset_loader.py        # PyTorch Dataset loader for mounted client partitions
│   │   ├── client_identity.py       # Resolves client identity from env & YAML configs
│   │   ├── metrics.py               # Accuracy calculation & psutil CPU/RAM metrics
│   │   └── event_publisher.py       # Asynchronous non-blocking client telemetry publisher
│   ├── Dockerfile                   # Python 3.11-slim client image
│   └── requirements.txt             # Client dependencies
│
├── monitoring/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, lifespan, CORS, static mounts
│   │   ├── api/
│   │   │   ├── routes.py            # Health, experiment status, events, internal event ingestion
│   │   │   ├── clients.py           # Client status and detailed telemetry endpoints
│   │   │   ├── rounds.py            # Round lifecycle stage endpoints
│   │   │   └── metrics.py           # Global metrics and client loss history endpoints
│   │   ├── websocket/
│   │   │   └── manager.py           # ConnectionManager for WebSocket broadcast & heartbeats
│   │   ├── database/
│   │   │   ├── models.py            # SQLAlchemy SQLite models with database indexes
│   │   │   ├── database.py          # SQLite engine & thread-safe sessionmaker
│   │   │   └── repository.py        # Idempotent CRUD operations
│   │   ├── services/
│   │   │   ├── event_processor.py   # State machine updating SQLite & broadcasting via WS
│   │   │   └── system_monitor.py    # Periodic background host resource monitor (psutil)
│   │   └── dashboard/
│   │       ├── index.html           # Dark-mode technical FL monitoring console
│   │       ├── style.css            # Responsive dashboard styles and status badges
│   │       └── dashboard.js         # WebSocket client with Chart.js curves & modal dialogs
│   ├── Dockerfile                   # Python 3.11-slim monitoring service image
│   └── requirements.txt             # Monitoring dependencies
│
├── datasets/
│   ├── original/cifar10/
│   │   ├── train/data.pt            # Clean training set (50,000 samples)
│   │   └── test/data.pt             # Isolated test set for global evaluation (10,000 samples)
│   ├── partitions/cifar10/
│   │   └── iid/clients_10_seed_42/  # Pre-generated deterministic client partitions
│   │       ├── client_001/data.pt
│   │       ├── client_002/data.pt
│   │       ├── client_003/data.pt
│   │       ├── manifest.json
│   │       ├── partition_config.yaml
│   │       └── statistics.csv
│   └── metadata/
│       └── dataset_registry.json    # SHA-256 checksums and verification logs
│
├── experiments/
│   └── default.yaml                 # Master experiment configuration
│
├── shared/
│   ├── schemas/
│   │   └── events.py                # Pydantic v2 event schemas shared across all tiers
│   └── configs/
│       └── client_config.yaml       # Default client configuration
│
├── logs/                            # Persistent log mounts for server, monitor, and clients
├── models/checkpoints/              # Saved PyTorch global model checkpoints (.pt)
├── docker-compose.yml               # Multi-container orchestration specification
├── run_local_experiment.py          # Standalone Python runner for full multi-process experiment
├── requirements.txt                 # Master pinned project requirements
└── README.md
```

---

## 3. Model Architecture & Federated Learning Dynamics

The system trains a lightweight, high-capacity Convolutional Neural Network (**`CIFAR10CNN`**) implemented in PyTorch, specifically balanced for CPU-based multi-container federated training.

### 3.1 Neural Network Architecture (`client/app/model.py`)

The network processes 3-channel RGB images $(3 \times 32 \times 32)$ and maps them to class probability logits across the 10 CIFAR-10 target categories:

```text
Input Tensor: (B, 3, 32, 32)
   │
   ├── [Block 1]
   │    ├── Conv2d(3 -> 32, kernel=3, padding=1) + BatchNorm2d(32) + ReLU
   │    ├── Conv2d(32 -> 64, kernel=3, padding=1) + BatchNorm2d(64) + ReLU
   │    └── MaxPool2d(kernel=2, stride=2)   ──► Output: (B, 64, 16, 16)
   │
   ├── [Block 2]
   │    ├── Conv2d(64 -> 128, kernel=3, padding=1) + BatchNorm2d(128) + ReLU
   │    └── MaxPool2d(kernel=2, stride=2)   ──► Output: (B, 128, 8, 8)
   │
   └── [Classifier Head]
        ├── Flatten(1)                      ──► Output: (B, 8192)
        ├── Linear(8192 -> 256) + ReLU
        ├── Dropout(p=0.25)
        └── Linear(256 -> 10)               ──► Output: (B, 10) logits
```

### 3.2 Detailed Layer & Trainable Parameter Breakdown

| Layer Name | Type | Input Shape | Output Shape | Kernel / Stride | Trainable Parameters | Parameter Formula |
|---|---|---|---|---|---|---|
| `conv1` | `Conv2d` | `(B, 3, 32, 32)` | `(B, 32, 32, 32)` | $3 \times 3$, pad 1 | **896** | $(3 \times 3 \times 3 \times 32) + 32$ |
| `bn1` | `BatchNorm2d` | `(B, 32, 32, 32)` | `(B, 32, 32, 32)` | - | **64** | $2 \times 32$ ($\gamma, \beta$) |
| `conv2` | `Conv2d` | `(B, 32, 32, 32)` | `(B, 64, 32, 32)` | $3 \times 3$, pad 1 | **18,496** | $(32 \times 3 \times 3 \times 64) + 64$ |
| `bn2` | `BatchNorm2d` | `(B, 64, 32, 32)` | `(B, 64, 32, 32)` | - | **128** | $2 \times 64$ ($\gamma, \beta$) |
| `pool1` | `MaxPool2d` | `(B, 64, 32, 32)` | `(B, 64, 16, 16)` | $2 \times 2$, stride 2 | **0** | Non-parametric |
| `conv3` | `Conv2d` | `(B, 64, 16, 16)` | `(B, 128, 16, 16)`| $3 \times 3$, pad 1 | **73,856** | $(64 \times 3 \times 3 \times 128) + 128$ |
| `bn3` | `BatchNorm2d` | `(B, 128, 16, 16)`| `(B, 128, 16, 16)`| - | **256** | $2 \times 128$ ($\gamma, \beta$) |
| `pool2` | `MaxPool2d` | `(B, 128, 16, 16)`| `(B, 128, 8, 8)`  | $2 \times 2$, stride 2 | **0** | Non-parametric |
| `flatten` | `Flatten` | `(B, 128, 8, 8)`  | `(B, 8192)`       | - | **0** | Reshaping operation |
| `fc1` | `Linear` | `(B, 8192)` | `(B, 256)` | - | **2,097,408** | $(8192 \times 256) + 256$ |
| `dropout` | `Dropout(0.25)`| `(B, 256)` | `(B, 256)` | - | **0** | Random zeroing ($p=0.25$) |
| `fc2` | `Linear` | `(B, 256)` | `(B, 10)` | - | **2,570** | $(256 \times 10) + 10$ |
| **Total** | - | - | - | - | **2,193,674** | **~2.19 Million Parameters** |

*Note: Non-trainable running mean and variance tracking buffers in BatchNorm layers add an additional 448 states.*

### 3.3 Loss Function & Local Optimizer

1. **Criterion**: Multi-Class Cross-Entropy Loss:
   $$\mathcal{L}_{CE}(y, \hat{y}) = -\sum_{c=1}^{10} y_c \log \left( \frac{e^{\hat{y}_c}}{\sum_{j=1}^{10} e^{\hat{y}_j}} \right)$$
2. **Optimizer**: 
   - **Adam** (default): Initial learning rate $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
   - **SGD** (configurable): Learning rate $\eta = 0.01$, Nesterov momentum $\mu = 0.9$.
3. **CPU Execution Optimization**: To avoid thread contention on multi-core host systems running multiple concurrent clients, PyTorch CPU execution threads are bounded per client process via `torch.set_num_threads(2)`.

### 3.4 Local Client Training Loop (`client/app/trainer.py`)

In each federated training round:
1. **Weight Synchronization**: The Flower `ClientApp` receives serialized global weights from the server via gRPC and applies them strictly to the local model:
   ```python
   set_model_parameters(model, parameters)
   ```
2. **Local Gradient Descent**: The client iterates over its mounted local partition for `LOCAL_EPOCHS` (default: 2 epochs) with a batch size of `BATCH_SIZE` (default: 32):
   - Computes forward pass predictions $\hat{y} = \text{model}(x)$.
   - Computes backpropagation gradients $\nabla_\theta \mathcal{L}_{CE}$.
   - Steps the local optimizer to update local parameters $\theta^{(k)}$.
3. **Telemetry & Resource Sampling**:
   - Calculates local epoch loss and top-1 accuracy.
   - Samples real-time process CPU utilization percentage and RSS RAM (MB) using `psutil`.
   - Packages metric payload and extracts updated weights as NumPy arrays:
     ```python
     parameters = get_model_parameters(model)
     ```
4. **Asynchronous Non-Blocking Reporting**: Enqueues a telemetry payload to a background daemon thread that posts to `http://server:8000/api/internal/events` (or `http://127.0.0.1:8000/api/internal/events` locally) without blocking gRPC responses.

### 3.5 Global Federated Aggregation (`server/app/strategy.py`)

The server executes the **`MonitoredFedAvg`** strategy (extending McMahan et al.'s Federated Averaging algorithm):

1. **Update Collection & Anomaly Filtering**:
   - Validates client parameter vectors for non-finite values (`NaN`, `Inf`) and dimension mismatches.
   - Rejects corrupted or malicious updates before aggregation.
2. **Weighted Model Parameter Averaging**:
   The new global parameter vector $w_{t+1}$ is computed as the sample-size-weighted average of all participating client updates:
   $$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{N} w_k^{t+1}, \quad \text{where } N = \sum_{k=1}^{K} n_k$$
   - $K$: Number of participating clients in round $t$ ($K \ge \text{min\_fit\_clients}$).
   - $n_k$: Number of local training examples processed by client $k$.
   - $w_k^{t+1}$: Model weight vector returned by client $k$.
3. **Centralized Server Evaluation**:
   - The aggregated global model is tested against the pristine, isolated CIFAR-10 test set ($10,000$ samples) located at `datasets/original/cifar10/test/data.pt`.
   - Computes global test loss and global test accuracy.
4. **Checkpointing & Lifecycle Broadcast**:
   - Saves checkpoint `.pt` file to `models/checkpoints/` containing `round`, `state_dict`, and evaluation metrics.
   - Emits structured telemetry events (`ROUND_COMPLETED`, `STAGE_UPDATED`) to the FastAPI monitor for real-time WebSocket dashboard broadcast.

---

## 4. Prerequisites

- **OS**: Windows 11 (64-bit), Linux, or macOS.
- **Python**: Python 3.11 (Python 3.11.0+ recommended).
- **RAM**: Minimum 8 GB (16 GB recommended).
- **Docker**: Docker Desktop with WSL2 backend (if deploying via containers).

---

## 5. Windows 11 + WSL2 + Docker Desktop Setup

To run via Docker containers on Windows 11:

1. **Enable Virtualization & WSL2**:
   - Ensure **Intel Virtualization Technology (VT-x)** or **AMD-V / SVM** is enabled in your machine's BIOS/UEFI.
   - Open PowerShell as Administrator and enable the Virtual Machine Platform:
     ```powershell
     dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
     wsl.exe --update
     wsl.exe --set-default-version 2
     ```
   - Restart your machine if prompted.

2. **Install Docker Desktop**:
   - Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
   - In Docker Desktop Settings:
     - Check **"Use the WSL 2 based engine"** under *General*.
     - Ensure your WSL distribution is enabled under *Resources &rarr; WSL Integration*.
   - Start Docker Desktop and verify the daemon is running:
     ```powershell
     docker --version
     docker compose version
     ```

---

## 6. Installation & Local Setup

1. **Clone and navigate to repository**:
   ```powershell
   cd c:\Users\rrpra\Documents\Github\fl_framework_docker
   ```

2. **Set up Python 3.11 Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verify or Generate Dataset Partitions**:
   Ensure pre-generated CIFAR-10 partitions exist:
   ```powershell
   python -c "from dataset_manager.partition_manager import PartitionManager; PartitionManager().generate_partitions('iid', 10, 42)"
   ```

4. **Run Automated Test Suite**:
   Verify all 52 unit and integration tests across server, client, monitoring, and dataset manager:
   ```powershell
   $env:PYTHONPATH="."
   python -m pytest tests -v
   ```

---

## 7. Docker Orchestration & Command Reference

Docker Compose provides declarative orchestration of the federated learning ecosystem inside an isolated bridge network (`fl_network`). The central **`fl_server`** container serves both Flower FL gRPC on port `8080` and the interactive FastAPI Monitoring Dashboard on port `8000`, while client containers (`fl_client_001` through `fl_client_010`) run isolated PyTorch local training loops.

---

### 7.1 Quick Commands Cheat Sheet

| Action | PowerShell / Windows Command | Linux / macOS Bash Command |
|---|---|---|
| **Build Images** | `docker compose build` | `docker compose build` |
| **Start Server & Web Monitor** | `docker compose up -d server` | `docker compose up -d server` |
| **Open Web Monitor** | Open browser to `http://localhost:8000` | Open browser to `http://localhost:8000` |
| **Run 2 Clients** | `docker compose up --force-recreate client_001 client_002` | `docker compose up --force-recreate client_001 client_002` |
| **Run 3 Clients** | `docker compose up --force-recreate client_001 client_002 client_003` | `docker compose up --force-recreate client_001 client_002 client_003` |
| **Run 5 Clients** | `docker compose up --force-recreate client_001 client_002 client_003 client_004 client_005` | `docker compose up --force-recreate client_001 client_002 client_003 client_004 client_005` |
| **Run 10 Clients** | `docker compose up --force-recreate client_001 client_002 client_003 client_004 client_005 client_006 client_007 client_008 client_009 client_010` | `docker compose up --force-recreate client_001 client_002 client_003 client_004 client_005 client_006 client_007 client_008 client_009 client_010` |
| **Run Clients in Background (-d)** | `docker compose up -d --force-recreate client_001 client_002` | `docker compose up -d --force-recreate client_001 client_002` |
| **Stream Server Logs** | `docker compose logs -f server` | `docker compose logs -f server` |
| **Stream Client Logs** | `docker compose logs -f client_001` | `docker compose logs -f client_001` |
| **Clean Up Finished Clients** | `docker compose rm -f client_001 client_002 client_003 client_004 client_005` | `docker compose rm -f client_001 client_002 client_003 client_004 client_005` |
| **Stop All Containers** | `docker compose stop` | `docker compose stop` |
| **Teardown Containers & Network** | `docker compose down` | `docker compose down` |
| **Complete Teardown (+ Volumes)** | `docker compose down -v` | `docker compose down -v` |

> [!IMPORTANT]
> **Always include `--force-recreate` when starting clients!**  
> In Docker, if a container has stopped from an earlier run and the Docker bridge network was refreshed, attempting to start the stopped container can fail with `network not found`. Passing `--force-recreate` guarantees fresh containers attached to the active network every time.

---

### 7.2 Running the Unified Server & Monitoring Console

The unified central server container (`fl_server`) concurrently hosts:
- **Port 8080 (TCP / gRPC)**: Flower SuperLink endpoint for client synchronization and model distribution.
- **Port 8000 (HTTP / WebSockets)**: Interactive dark-mode dashboard, REST API, and telemetry streaming.
- **Persistent Volumes**: `./logs` (server log files) and `./results` (SQLite DB and multi-format run reports).
- **Continuous Mode**: Once training rounds conclude and reports are generated, the server automatically keeps its gRPC listener on port `8080` open so subsequent client runs can connect without restarting the container.

#### Start the Server:
```bash
docker compose up -d server
```

#### Check Server Health & Readiness:
```bash
# Verify health endpoint (returns 200 OK {"status": "healthy"})
curl -s http://localhost:8000/api/health

# Check container state
docker compose ps
```

#### Access Web Dashboard:
Open **`http://localhost:8000`** in any web browser to view live telemetry, loss/accuracy curves, volume server logs, and the run reports audit tab.

---

### 7.3 Running Different Numbers of Clients (2, 3, 5, 10 Clients)

The [`docker-compose.yml`](file:///docker-compose.yml) file includes pre-configured services for **`client_001` through `client_010`**, each mapped to its isolated partition in `./datasets/partitions/` and independent log volume in `./logs/clients/`.

#### Controlling Client Participation via `.env`:
Before launching a run with a specific client count, configure how many clients Flower should wait for before starting Round 1. Set these in [`.env`](file:///.env):
```env
NUM_ROUNDS=3              # Number of federated rounds
MIN_FIT_CLIENTS=5         # Minimum clients required to train per round
MIN_AVAILABLE_CLIENTS=5   # Server will wait until 5 clients connect before starting Round 1
```

#### Example 1: Running with 2 Clients (Baseline)
```bash
docker compose up --force-recreate client_001 client_002
```

#### Example 2: Running with 3 Clients
```bash
docker compose up --force-recreate client_001 client_002 client_003
```

#### Example 3: Running with 5 Clients (Simultaneous)
Ensure `.env` has `MIN_AVAILABLE_CLIENTS=5` and `MIN_FIT_CLIENTS=5`, then run:
```bash
docker compose up --force-recreate client_001 client_002 client_003 client_004 client_005
```

#### Example 4: Running with All 10 Clients (Full Partition Suite)
Ensure `.env` has `MIN_AVAILABLE_CLIENTS=10` and `MIN_FIT_CLIENTS=10`, then run:
```bash
docker compose up --force-recreate client_001 client_002 client_003 client_004 client_005 client_006 client_007 client_008 client_009 client_010
```

#### Inline Shell Execution (Overriding Client Counts Dynamically):
You can override participation parameters directly on the command line without modifying `.env`:

```powershell
# Windows PowerShell (Wait for 5 clients)
$env:MIN_AVAILABLE_CLIENTS="5"; $env:MIN_FIT_CLIENTS="5"; docker compose up -d server
docker compose up --force-recreate client_001 client_002 client_003 client_004 client_005
```

```bash
# Linux / macOS Bash (Wait for 5 clients)
MIN_AVAILABLE_CLIENTS=5 MIN_FIT_CLIENTS=5 docker compose up -d server
docker compose up --force-recreate client_001 client_002 client_003 client_004 client_005
```

---

### 7.4 Running External Clients (LAN, Wi-Fi, or Global IP)

The framework supports distributed setups where the Flower server runs on one machine and federated clients run on other physical computers, edge devices, or cloud VMs.

#### Scenario 1: Same Computer via Docker Compose (Default)
In `.env`:
```env
SERVER_ADDRESS=server:8080
MONITORING_URL=http://server:8000
```
Clients connect over Docker's internal DNS without traversing host firewalls.

#### Scenario 2: External Client on Another PC (Same LAN / Wi-Fi)
1. **On the Server Host (Windows Firewall)**:  
   Open **PowerShell as Administrator** and add an inbound firewall rule for ports 8080 and 8000:
   ```powershell
   New-NetFirewallRule -DisplayName "Flower FL Server (8080 & 8000)" -Direction Inbound -LocalPort 8080,8000 -Protocol TCP -Action Allow -Profile Any
   ```
2. **Find Server IP**: Run `ipconfig` (Windows) or `hostname -I` (Linux) to find the server's LAN IP (e.g., `10.135.73.182`).
3. **On the External Client PC**:  
   Run the client Docker image pointing to the server's LAN IP:
   ```bash
   docker run --rm -it \
     -e SERVER_ADDRESS="10.135.73.182:8080" \
     -e MONITORING_URL="http://10.135.73.182:8000" \
     -e CLIENT_ID="client_001" \
     -v /path/to/partition:/app/data:ro \
     fl_client:latest
   ```

#### Scenario 3: Global IP / Over the Public Internet
Since LAN IPs (`10.x.x.x` or `192.168.x.x`) are not routable across the public internet:
1. **Using ngrok (Zero-Config Reverse Tunnel - Recommended)**:
   On the server machine, expose port 8080:
   ```bash
   ngrok tcp 8080
   ```
   ngrok provides a public forwarding endpoint (e.g. `0.tcp.ngrok.io:12345`). Set `SERVER_ADDRESS="0.tcp.ngrok.io:12345"` on any remote client anywhere in the world!
2. **Using Tailscale / WireGuard**:
   Install Tailscale on both server and client devices; use the server's 100.x.x.x Tailscale IP with zero router configuration.
3. **Router Port Forwarding**:
   Forward external TCP ports 8080 and 8000 on your router to the server's internal LAN IP, then use your public WAN IP.

---

### 7.5 Container Inspection, Logs & Top Resources

Check state, health status, and port bindings:
```bash
docker compose ps
```
Inspect process tree and resource consumption inside active containers:
```bash
docker compose top
```
Follow logs in real time:
```bash
# Follow unified server logs
docker compose logs -f server

# Follow client logs
docker compose logs -f client_001 client_002
```

---

### 7.6 Stopping, Cleaning Up & Recreating Containers

```bash
# Stop all containers while preserving state and database
docker compose stop

# Restart a specific service
docker compose restart server
docker compose restart client_001

# Remove stopped client containers
docker compose rm -f client_001 client_002 client_003 client_004 client_005

# Stop and remove all containers and network
docker compose down

# Complete teardown including volumes (resets SQLite DB and metrics)
docker compose down -v
```

### 7.7 Docker Hub Image Registry
Container images for the framework components:
- **Flower Central FL Server & Monitor**: `ashutosh310106/fl_server:latest`
- **Federated Client Node**: `ashutosh310106/fl_client:latest`

---

## 8. Terminal & CLI Orchestration Reference

For rapid experimentation, debugging, or running on systems without Docker, the framework provides both an automated multi-process runner and individual CLI execution.

### 8.1 Automated Single-Command Runner (`run_local_experiment.py`)

The included runner automatically manages the lifecycle of the FastAPI monitor, Flower server, and 3 concurrent client processes, verifying all training rounds to completion:

```powershell
# Windows PowerShell
python run_local_experiment.py
```
```bash
# Linux / macOS Bash
python3 run_local_experiment.py
```

### 8.2 Manual Multi-Terminal Orchestration

To run each service in its own dedicated terminal window for step-by-step debugging:

#### Terminal 1 — FastAPI Monitoring Service
```powershell
# Windows PowerShell
$env:PYTHONPATH="."
uvicorn monitoring.app.main:app --host 0.0.0.0 --port 8000 --reload
```
```bash
# Linux / macOS Bash
export PYTHONPATH="."
uvicorn monitoring.app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2 — Flower FL Server
```powershell
# Windows PowerShell
$env:PYTHONPATH="."
$env:SERVER_ADDRESS="127.0.0.1:8080"
$env:MONITORING_URL="http://127.0.0.1:8000"
$env:NUM_ROUNDS="3"
$env:MIN_FIT_CLIENTS="3"
$env:MIN_AVAILABLE_CLIENTS="3"
python -m server.app.server_app
```
```bash
# Linux / macOS Bash
export PYTHONPATH="."
export SERVER_ADDRESS="127.0.0.1:8080"
export MONITORING_URL="http://127.0.0.1:8000"
export NUM_ROUNDS="3"
export MIN_FIT_CLIENTS="3"
export MIN_AVAILABLE_CLIENTS="3"
python3 -m server.app.server_app
```

#### Terminals 3, 4, 5 — Federated Clients

**Client 1 (`client_001`):**
```powershell
# Windows PowerShell
$env:PYTHONPATH="."
$env:CLIENT_ID="client_001"
$env:SERVER_ADDRESS="127.0.0.1:8080"
$env:MONITORING_URL="http://127.0.0.1:8000"
$env:PARTITION_PATH="datasets/partitions/cifar10/iid/clients_10_seed_42/client_001"
python -m client.app.client_app
```
```bash
# Linux / macOS Bash
export PYTHONPATH="."
export CLIENT_ID="client_001"
export SERVER_ADDRESS="127.0.0.1:8080"
export MONITORING_URL="http://127.0.0.1:8000"
export PARTITION_PATH="datasets/partitions/cifar10/iid/clients_10_seed_42/client_001"
python3 -m client.app.client_app
```

**Client 2 (`client_002`):**
```powershell
# Windows PowerShell
$env:PYTHONPATH="."
$env:CLIENT_ID="client_002"
$env:SERVER_ADDRESS="127.0.0.1:8080"
$env:MONITORING_URL="http://127.0.0.1:8000"
$env:PARTITION_PATH="datasets/partitions/cifar10/iid/clients_10_seed_42/client_002"
python -m client.app.client_app
```
```bash
# Linux / macOS Bash
export PYTHONPATH="."
export CLIENT_ID="client_002"
export SERVER_ADDRESS="127.0.0.1:8080"
export MONITORING_URL="http://127.0.0.1:8000"
export PARTITION_PATH="datasets/partitions/cifar10/iid/clients_10_seed_42/client_002"
python3 -m client.app.client_app
```

**Client 3 (`client_003`):**
```powershell
# Windows PowerShell
$env:PYTHONPATH="."
$env:CLIENT_ID="client_003"
$env:SERVER_ADDRESS="127.0.0.1:8080"
$env:MONITORING_URL="http://127.0.0.1:8000"
$env:PARTITION_PATH="datasets/partitions/cifar10/iid/clients_10_seed_42/client_003"
python -m client.app.client_app
```
```bash
# Linux / macOS Bash
export PYTHONPATH="."
export CLIENT_ID="client_003"
export SERVER_ADDRESS="127.0.0.1:8080"
export MONITORING_URL="http://127.0.0.1:8000"
export PARTITION_PATH="datasets/partitions/cifar10/iid/clients_10_seed_42/client_003"
python3 -m client.app.client_app
```

### 8.3 Dataset Management & Partitioning CLI

Download clean CIFAR-10 data and generate heterogeneous client splits:

```bash
# 1. Download original clean CIFAR-10 training & test sets
python -c "from dataset_manager.dataset_loader import download_cifar10; download_cifar10()"

# 2. Generate IID partitions (10 clients, seed 42)
python -c "from dataset_manager.partition_manager import PartitionManager; PartitionManager().generate_partitions('iid', 10, 42)"

# 3. Generate Non-IID Dirichlet distribution (alpha = 0.5)
python -c "from dataset_manager.partition_manager import PartitionManager; PartitionManager().generate_partitions('dirichlet', 10, 42, {'alpha': 0.5})"

# 4. Generate Non-IID Label Skew (2 classes per client)
python -c "from dataset_manager.partition_manager import PartitionManager; PartitionManager().generate_partitions('label_skew', 10, 42, {'classes_per_client': 2})"

# 5. Launch interactive Dataset Explorer in Streamlit
streamlit run dataset_manager/app.py
```

### 8.4 REST API & Health Probing via CLI

Query live telemetry directly from the command line:

```bash
# Probe monitoring service health
curl -s http://localhost:8000/api/health | jq

# Get current experiment state & summary
curl -s http://localhost:8000/api/experiment/current | jq

# List registered federated clients and metrics
curl -s http://localhost:8000/api/clients | jq

# Inspect global accuracy and loss progression
curl -s http://localhost:8000/api/metrics/global | jq
```

---

## 9. Live Monitoring Dashboard

Access the browser dashboard at:
```text
http://localhost:8000
```

### Dashboard Capabilities:
- **Experiment Status Bar**: Shows live experiment ID, current state (`IDLE`, `RUNNING`, `COMPLETED`), and active aggregation strategy.
- **Round Progress Bar**: Shows current round out of total configured rounds.
- **Global Metrics Chart**: Interactive Chart.js showing global test accuracy (%) and loss progression across rounds.
- **Client Loss Comparison Chart**: Per-client training loss curves over rounds.
- **Round Lifecycle Tracker**: Displays 5 distinct round stages with status indicators:
  1. `Model Distribution`
  2. `Client Training`
  3. `Update Collection`
  4. `Aggregation`
  5. `Global Evaluation`
- **Client Telemetry Table**: Real-time table displaying client ID, role, status badges (`Registered`, `Connected`, `Training`, `Waiting`, `Completed`), dataset size, loss, accuracy, and process CPU/RAM.
- **Client Detailed Modal**: Click **"View"** on any client row to inspect hardware footprint, local dataset partition, and full historical metrics table.
- **Event Audit Feed**: Chronological live event feed filterable by severity (`INFO`, `SUCCESS`, `WARNING`, `ERROR`) and client ID.

---

## 10. REST API Documentation

The monitoring service exposes the following OpenAPI endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check probe for Docker and monitoring |
| `GET` | `/api/experiment/current` | Metadata, status, and config of active experiment |
| `GET` | `/api/experiments` | Historical list of all experiments |
| `GET` | `/api/clients` | Summary list of all registered clients and live status |
| `GET` | `/api/clients/{client_id}` | Detailed telemetry and training history for a client |
| `GET` | `/api/rounds` | Status and duration of all federated rounds and stages |
| `GET` | `/api/rounds/{round_id}` | Detailed stage progress and events for a round |
| `GET` | `/api/metrics/global` | Global test evaluation metrics across rounds |
| `GET` | `/api/metrics/clients/{client_id}` | Local loss and accuracy history for a specific client |
| `GET` | `/api/system/resources` | CPU and RAM utilization snapshots |
| `GET` | `/api/events` | Chronological event audit feed with filtering |
| `GET` | `/api/reports` | List historical run reports saved in results volume |
| `GET` | `/api/reports/latest` | Retrieve the most recent experiment run report |
| `GET` | `/api/reports/{run_id}` | Detailed report payload (Markdown/JSON/CSV summary) |
| `GET` | `/api/logs/{service}` | Tail service container logs from the persistent volume |
| `POST` | `/api/internal/events` | Internal event ingestion route (returns HTTP 202) |
| `WS` | `/ws/live` | WebSocket connection for real-time dashboard events |

---

## 11. Inspecting Individual Clients

### Via Dashboard:
In the **Federated Clients Telemetry** table, click the **"View"** button for any client. A modal opens displaying:
- Role & identity
- Container/process status
- Partition sample count
- Local training loss & accuracy
- CPU and RAM footprint
- Historical per-round metrics

### Via REST API:
```bash
curl -s http://localhost:8000/api/clients/client_001 | jq
```

---

## 12. Viewing Round Lifecycle & History

Each round undergoes 5 monitored stages with durations recorded upon completion:

```text
Round 1 (41.1s)
  1. Distribution       COMPLETED
  2. Client Training    COMPLETED
  3. Update Collection  COMPLETED
  4. Aggregation        COMPLETED
  5. Global Eval        COMPLETED
```

Retrieve round details via REST API:
```bash
curl -s http://localhost:8000/api/rounds/1 | jq
```

---

## 13. Stopping & Restarting Containers

### Stop all containers:
```bash
docker compose down
```

### Stop without removing container state:
```bash
docker compose stop
```

### Restart a specific client container:
```bash
docker compose restart client_001
```

### Restart the unified server and web monitor:
```bash
docker compose restart server
```
*The SQLite database in `./results/monitoring.db` persists state across container restarts.*

---

## 14. Scaling to Additional Clients & Participation Settings

The framework comes pre-configured with **10 ready-to-run client services** (`client_001` through `client_010`) in [`docker-compose.yml`](file:///docker-compose.yml), each connected to its respective partition and log volume.

### 14.1 Configuring Participation Counts in `.env`
You can dynamically control how many clients Flower requires for training:

```env
# Total number of federated rounds to run:
NUM_ROUNDS=3

# Number of clients that must train and return updates per round:
MIN_FIT_CLIENTS=5

# Minimum clients connected to the server before Round 1 begins:
MIN_AVAILABLE_CLIENTS=5
```

- When `MIN_AVAILABLE_CLIENTS` is met, Flower triggers `configure_fit()`.
- If you launch 5 clients while `MIN_AVAILABLE_CLIENTS=2`, Flower will start Round 1 as soon as the first 2 clients connect, and the other 3 will join in Round 2.
- To ensure **all 5 clients participate from Round 1**, always set `MIN_AVAILABLE_CLIENTS=5` and `MIN_FIT_CLIENTS=5`.

### 14.2 Scaling Beyond 10 Clients (e.g., Adding `client_011`)
To scale beyond 10 clients:
1. Generate the partitions (e.g. for 20 clients):
   ```bash
   python -c "from dataset_manager.partition_manager import PartitionManager; PartitionManager().generate_partitions('iid', 20, 42)"
   ```
2. Append a new service block in `docker-compose.yml`:
   ```yaml
   client_011:
     build:
       context: .
       dockerfile: client/Dockerfile
     image: fl_client:latest
     container_name: fl_client_011
     depends_on:
       server:
         condition: service_started
     environment:
       - CLIENT_ID=client_011
       - CLIENT_ROLE=honest
       - PARTITION_PATH=/app/data
       - SERVER_ADDRESS=${SERVER_ADDRESS:-server:8080}
       - MONITORING_URL=${MONITORING_URL:-http://server:8000}
       - LOCAL_EPOCHS=2
       - BATCH_SIZE=32
       - LEARNING_RATE=0.001
       - LOG_DIR=/app/logs
       - PYTHONUNBUFFERED=1
     volumes:
       - ${PARTITION_DIR:-./datasets/partitions/cifar10/iid/clients_10_seed_42}/client_011:/app/data:ro
       - ./logs/clients/client_011:/app/logs
     networks:
       - fl_network
     deploy:
       resources:
         limits:
           cpus: "1.0"
           memory: 1G
   ```
3. Launch the new client:
   ```bash
   docker compose up --force-recreate client_011
   ```

---

## 15. Dataset Partitions & Management (Changing Datasets)

The framework includes an enterprise data management suite supporting pre-generated CIFAR-10 partitions and deterministic generators for evaluating federated learning algorithms under extreme non-IID conditions.

- **Clean Training Set**: 50,000 samples stored in `datasets/original/cifar10/train/data.pt`.
- **Isolated Test Set**: 10,000 samples in `datasets/original/cifar10/test/data.pt`, reserved exclusively for central server evaluation.
- **Client Partition Format**: Stored as PyTorch serialized dictionaries (`data.pt`):
  ```python
  {
      "images": torch.Tensor(N, 3, 32, 32), # uint8 (0-255)
      "labels": torch.Tensor(N,),           # int64 (0-9)
      "sample_indices": torch.Tensor(N,)    # int64
  }
  ```

---

### 15.1 Switching Active Dataset Partitions via `.env`

You can switch the active dataset partition strategy across all 10 client containers in **one line** without touching `docker-compose.yml`.

Edit the `PARTITION_DIR` variable in [`.env`](file:///.env):

```env
# 1. Standard IID (Default - 10 balanced clients, uniform label distribution)
PARTITION_DIR=./datasets/partitions/cifar10/iid/clients_10_seed_42

# 2. Non-IID Dirichlet (Alpha = 0.5 - realistic class distribution skew)
# PARTITION_DIR=./datasets/partitions/cifar10/dirichlet/clients_10_alpha_0_5_seed_42

# 3. Non-IID Label Skew (2 classes per client - extreme non-IID partition)
# PARTITION_DIR=./datasets/partitions/cifar10/label_skew/clients_10_classes_2_seed_42

# 4. Non-IID Quantity Skew (Sample volume imbalance between 1,000 and 9,000 samples)
# PARTITION_DIR=./datasets/partitions/cifar10/quantity_skew/clients_10_min1000_max9000_seed_42
```

After modifying `PARTITION_DIR`, start the clients:
```bash
docker compose up --force-recreate client_001 client_002 client_003
```
Each container automatically mounts its assigned partition directory from the selected distribution strategy.

---

### 15.2 Generating Heterogeneous Partitions via CLI

The built-in Partition Manager allows you to generate new partition configurations with custom client counts, random seeds, and statistical skews:

```bash
# 1. Download clean CIFAR-10 training & test datasets
python -c "from dataset_manager.dataset_loader import download_cifar10; download_cifar10()"

# 2. Generate IID partitions (10 clients, seed 42)
python -c "from dataset_manager.partition_manager import PartitionManager; PartitionManager().generate_partitions('iid', 10, 42)"

# 3. Generate Non-IID Dirichlet distribution with custom alpha
# (alpha=0.1 = extreme class imbalance; alpha=0.5 = moderate skew; alpha=1.0 = mild skew)
python -c "from dataset_manager.partition_manager import PartitionManager; PartitionManager().generate_partitions('dirichlet', 10, 42, {'alpha': 0.1})"

# 4. Generate Non-IID Label Skew (e.g. 2 or 3 classes per client)
python -c "from dataset_manager.partition_manager import PartitionManager; PartitionManager().generate_partitions('label_skew', 10, 42, {'classes_per_client': 2})"

# 5. Generate Non-IID Quantity Skew (custom minimum and maximum sample bounds)
python -c "from dataset_manager.partition_manager import PartitionManager; PartitionManager().generate_partitions('quantity_skew', 10, 42, {'min_samples': 500, 'max_samples': 7000})"
```

Each generated partition creates:
- `client_001/data.pt`, `client_002/data.pt`, ...
- `manifest.json`: SHA-256 integrity checksums for every client partition.
- `partition_config.yaml`: Exact reproducibility configuration and parameters.
- `statistics.csv`: Per-client sample counts and class distribution tables.

---

### 15.3 Interactive Dataset Explorer (Streamlit)

Visualize class distributions, sample volume heatmaps, and dataset skew interactively in your browser:

```bash
streamlit run dataset_manager/app.py
```
This opens the Dataset Explorer on `http://localhost:8501`, allowing visual side-by-side comparison between IID, Dirichlet, Label Skew, and Quantity Skew splits.

---

### 15.4 Adding a Custom Dataset (Format Specification)

To use your own custom dataset in place of CIFAR-10:
1. Ensure image tensors are shape `(N, C, H, W)` and labels are shape `(N,)`.
2. Format client partitions as serialized PyTorch dictionaries:
   ```python
   import torch
   from pathlib import Path

   # Save for client_001
   partition_dir = Path("datasets/partitions/custom_dataset/client_001")
   partition_dir.mkdir(parents=True, exist_ok=True)

   torch.save({
       "images": client_images,          # Tensor of shape (N, 3, 32, 32)
       "labels": client_labels,          # Tensor of shape (N,)
       "sample_indices": client_indices  # Tensor of shape (N,)
   }, partition_dir / "data.pt")
   ```
3. Set `PARTITION_DIR=./datasets/partitions/custom_dataset` in `.env`.
4. If image dimensions or classes differ from CIFAR-10 (3x32x32, 10 classes), update `server/app/model.py` and `experiments/default.yaml` accordingly.

---

## 16. Logs & Model Checkpoints

### Logs:
Logs are written to persistent directories mounted on the host:
- Server logs: `logs/server/`
- Monitor logs: `logs/monitoring/`
- Client logs: `logs/clients/<client_id>/`

### Model Checkpoints:
After each successful round aggregation, global model weights are saved to:
```text
models/checkpoints/cifar10_baseline_001_round_001.pt
models/checkpoints/cifar10_baseline_001_round_002.pt
models/checkpoints/cifar10_baseline_001_round_003.pt
```

Each checkpoint contains:
```python
{
    "round": 3,
    "experiment_id": "cifar10_baseline_001",
    "model_state_dict": model.state_dict(),
    "metrics": {"accuracy": 0.5886, ...},
    "timestamp": 1793456789.0
}
```

### Automated Run Reports:
At the conclusion of an experiment, the server automatically produces persistent multi-format audit reports saved in `results/run_reports/`:
- **Markdown Summary (`*_report.md`)**: Human-readable run summary detailing final accuracy, loss progression, client contributions, and round timestamps.
- **JSON Telemetry Report (`*_report.json`)**: Complete structured audit payload accessible programmatically via `GET /api/reports/latest`.
- **Per-Round Metrics CSV (`*_metrics.csv`)**: Tabular time-series for easy import into pandas, Excel, or plotting frameworks.

---

## 17. Troubleshooting Common Errors

### 1. `docker: The term 'docker' is not recognized` (Windows)
- Ensure Docker Desktop is installed and running.
- Ensure `C:\Program Files\Docker\Docker\resources\bin` is added to your Windows PATH environment variable.
- In Docker Desktop, verify WSL2 integration is enabled.

### 2. `Wsl/Service/CreateInstance/CreateVm/HCS/HCS_E_HYPERV_NOT_INSTALLED`
- Virtualization is disabled in your machine's BIOS/UEFI. Reboot your PC, enter BIOS setup, and enable **Intel VT-x** or **AMD-V / SVM**.

### 3. Client Connection Refused: `Failed to connect to server:8080`
- The Flower server takes 1-2 seconds to initialize gRPC.
- The client application incorporates an automatic exponential backoff retry loop (up to 30 attempts). Verify that `server` is running and accessible on the Docker network `fl_network`.

### 4. Flower `start_server()` deprecation notice
- Flower 1.x logs a warning indicating that CLI `flower-superlink` is preferred in production deployments.
- The code provides modern `ServerApp` and `ClientApp` components, compatible with both embedded gRPC runtimes and SuperLink/SuperNode architectures.

### 5. `SSL: CERTIFICATE_VERIFY_FAILED` when downloading CIFAR-10
- Use `kagglehub` or `certifi` bundle:
  ```powershell
  python -c "import kagglehub; kagglehub.dataset_download('pankrzysiu/cifar10-python')"
  ```

---

## License

Part of the Federated Learning Security Testing and Benchmarking Framework.  
Distributed under the MIT License.
