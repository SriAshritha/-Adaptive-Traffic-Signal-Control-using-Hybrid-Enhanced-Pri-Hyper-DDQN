# Pri-DDQN Benchmark Adaptation

This folder contains a benchmark-fair adaptation of:

`Pri-DDQN: learning adaptive traffic signal control strategy through a hybrid agent`

Implemented components:
- Double DQN
- priority-based dynamic replay
- power-function epsilon decay
- asynchronous target network updates
- state/reward-aware loss weighting

Benchmarks:
- original 2-action single-intersection setup
- 4-stage movement-based benchmark from `main_experiment_6action`

Run:

```powershell
C:\Users\USER\.conda\envs\traffic_dqn\python.exe Pri-DDQN\train.py
```
