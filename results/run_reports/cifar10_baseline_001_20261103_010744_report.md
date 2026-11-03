# Federated Learning Run Report: CIFAR10 Flower Baseline

**Experiment ID:** `cifar10_baseline_001_20261103_010744`  
**Status:** `COMPLETED`  
**Strategy:** `FedAvg`  
**Started:** 2026-11-03T01:07:45.342699+00:00  
**Ended:** 2026-11-03T01:13:53.405060+00:00  
**Configured Rounds:** 3 | **Completed Rounds:** 3  
**Final Accuracy:** `59.48%`  
**Final Loss:** `1.1372`  

## 1. Global Accuracy & Loss Progression

| Round | Global Loss | Global Accuracy (%) | Test Samples | Round Duration | Participating Clients |
| :---: | :---------: | :-----------------: | :----------: | :------------: | :-------------------: |
| Round 1 | 2.2531 | 22.01% | 10000 | 14.12s | 5 |
| Round 2 | 1.3608 | 52.33% | 10000 | 14.60s | 5 |
| Round 3 | 1.1372 | 59.48% | 10000 | 14.79s | 5 |

## 2. Client Training Metrics by Round

### Round 1 Clients
| Client ID | Samples | Local Loss | Local Accuracy | Training Time |
| :-------- | :-----: | :--------: | :------------: | :-----------: |
| `client_002` | 5000 | 1.8142 | 35.04% | 89.61s |
| `client_004` | 5000 | 1.8298 | 34.70% | 90.89s |
| `client_005` | 5000 | 1.8651 | 32.74% | 87.02s |
| `client_001` | 5000 | 1.8013 | 34.92% | 95.36s |
| `client_003` | 5000 | 1.8632 | 33.91% | 95.92s |

### Round 2 Clients
| Client ID | Samples | Local Loss | Local Accuracy | Training Time |
| :-------- | :-----: | :--------: | :------------: | :-----------: |
| `client_004` | 5000 | 1.5543 | 42.28% | 78.39s |
| `client_003` | 5000 | 1.6286 | 38.89% | 94.62s |
| `client_002` | 5000 | 1.6359 | 38.15% | 88.54s |
| `client_001` | 5000 | 1.5779 | 40.31% | 90.15s |
| `client_005` | 5000 | 1.6183 | 38.50% | 90.62s |

### Round 3 Clients
| Client ID | Samples | Local Loss | Local Accuracy | Training Time |
| :-------- | :-----: | :--------: | :------------: | :-----------: |
| `client_004` | 5000 | 1.3404 | 51.09% | 83.60s |
| `client_002` | 5000 | 1.3743 | 49.00% | 86.70s |
| `client_005` | 5000 | 1.3816 | 48.52% | 88.05s |
| `client_001` | 5000 | 1.3584 | 50.37% | 87.50s |
| `client_003` | 5000 | 1.3887 | 48.19% | 92.42s |

## 3. Saved Model Checkpoints

- `/app/models/checkpoints/cifar10_baseline_001_20261103_010744_round_001.pt`
- `/app/models/checkpoints/cifar10_baseline_001_20261103_010744_round_002.pt`
- `/app/models/checkpoints/cifar10_baseline_001_20261103_010744_round_003.pt`
