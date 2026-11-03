# Federated Learning Run Report: CIFAR10 Flower Baseline

**Experiment ID:** `cifar10_baseline_001_20261102_055124`  
**Status:** `COMPLETED`  
**Strategy:** `FedAvg`  
**Started:** 2026-11-02T05:51:24.899619+00:00  
**Ended:** 2026-11-02T05:59:43.653007+00:00  
**Configured Rounds:** 3 | **Completed Rounds:** 3  
**Final Accuracy:** `59.24%`  
**Final Loss:** `1.1478`  

## 1. Global Accuracy & Loss Progression

| Round | Global Loss | Global Accuracy (%) | Test Samples | Round Duration | Participating Clients |
| :---: | :---------: | :-----------------: | :----------: | :------------: | :-------------------: |
| Round 1 | 2.0192 | 30.41% | 10000 | 14.80s | 2 |
| Round 2 | 1.3876 | 49.57% | 10000 | 14.47s | 5 |
| Round 3 | 1.1478 | 59.24% | 10000 | 14.43s | 5 |

## 2. Client Training Metrics by Round

### Round 1 Clients
| Client ID | Samples | Local Loss | Local Accuracy | Training Time |
| :-------- | :-----: | :--------: | :------------: | :-----------: |
| `client_004` | 5000 | 1.8047 | 36.58% | 73.84s |
| `client_002` | 5000 | 1.8444 | 34.58% | 76.60s |

### Round 2 Clients
| Client ID | Samples | Local Loss | Local Accuracy | Training Time |
| :-------- | :-----: | :--------: | :------------: | :-----------: |
| `client_001` | 5000 | 1.5244 | 44.05% | 88.34s |
| `client_002` | 5000 | 1.5062 | 43.83% | 102.93s |
| `client_004` | 5000 | 1.4843 | 45.97% | 93.28s |
| `client_003` | 5000 | 1.5396 | 42.48% | 100.77s |
| `client_005` | 5000 | 1.5608 | 42.19% | 94.03s |

### Round 3 Clients
| Client ID | Samples | Local Loss | Local Accuracy | Training Time |
| :-------- | :-----: | :--------: | :------------: | :-----------: |
| `client_003` | 5000 | 1.3449 | 50.98% | 95.50s |
| `client_002` | 5000 | 1.3416 | 50.49% | 99.60s |
| `client_005` | 5000 | 1.3569 | 50.43% | 82.54s |
| `client_004` | 5000 | 1.3000 | 52.43% | 98.08s |
| `client_001` | 5000 | 1.3556 | 50.78% | 71.83s |

## 3. Saved Model Checkpoints

- `/app/models/checkpoints/cifar10_baseline_001_20261102_055124_round_001.pt`
- `/app/models/checkpoints/cifar10_baseline_001_20261102_055124_round_002.pt`
- `/app/models/checkpoints/cifar10_baseline_001_20261102_055124_round_003.pt`
