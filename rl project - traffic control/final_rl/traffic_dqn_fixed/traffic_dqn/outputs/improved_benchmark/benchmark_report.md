# Improved Agent Benchmark

| Agent | Reward | Std | Avg Queue | Avg Wait | Throughput | Phase Changes |
|---|---:|---:|---:|---:|---:|---:|
| Improved TD(0) | -58.706 | 2.445 | 0.467 | 3.905 | 5766.933 | 1869.167 |
| Improved SARSA | -58.571 | 2.455 | 0.475 | 3.951 | 5766.900 | 1807.067 |
| Improved Q-Learning | -58.427 | 2.412 | 0.475 | 3.932 | 5766.900 | 1810.267 |

## Reference Models

| Reference | Reward | Std | Avg Queue | Avg Wait | Throughput | Phase Changes |
|---|---:|---:|---:|---:|---:|---:|
| Pri-DDQN | -58.325 | 2.303 | 0.450 | 3.924 | 5767.000 | 1797.667 |
| MA-SAC | -65.511 | 2.527 | 1.000 | 4.746 | 5762.000 | 2103.000 |

## Metric Leaders

- Best reward: Pri-DDQN
- Lowest avg queue: Pri-DDQN
- Lowest avg wait: Improved TD(0)
- Highest throughput: Pri-DDQN

## Head-to-Head

### Improved TD(0)

| Reference | Better Reward | Better Queue | Better Wait | Better Throughput |
|---|---|---|---|---|
| Pri-DDQN | False | False | True | False |
| MA-SAC | True | True | True | True |

### Improved SARSA

| Reference | Better Reward | Better Queue | Better Wait | Better Throughput |
|---|---|---|---|---|
| Pri-DDQN | False | False | False | False |
| MA-SAC | True | True | True | True |

### Improved Q-Learning

| Reference | Better Reward | Better Queue | Better Wait | Better Throughput |
|---|---|---|---|---|
| Pri-DDQN | False | False | False | False |
| MA-SAC | True | True | True | True |
