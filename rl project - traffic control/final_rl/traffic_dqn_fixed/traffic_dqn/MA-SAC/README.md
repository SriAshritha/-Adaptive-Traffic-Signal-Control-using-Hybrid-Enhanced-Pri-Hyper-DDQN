# MA-SAC Benchmark Adaptation

This folder contains a benchmark-fair adaptation of the paper:

`Towards Multi-agent Reinforcement Learning based Traffic Signal Control through Spatio-temporal Hypergraphs`

Scope:
- Same 4-action benchmark as `main_experiment_6action`
- Same reward, horizon, and evaluation setting
- No SUMO deployment in this package
- MA-SAC style actor-critic with a spatio-temporal hypergraph-inspired centralized critic

Run:

```powershell
C:\Users\USER\.conda\envs\traffic-rl\python.exe MA-SAC\train.py
```
