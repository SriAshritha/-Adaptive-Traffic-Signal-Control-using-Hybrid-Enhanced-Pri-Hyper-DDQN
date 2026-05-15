# Methodology Note

## Repository audit
- Benchmark environment reused from `main_experiment_6action/environments/traffic_env.py`.
- Action space fixed to four movement stages: NS straight, NS left/U-turn, EW straight, EW left/U-turn.
- Reward, decision interval, yellow duration, horizon, and evaluation protocol were kept unchanged.
- Requested environment was `traffic-rl`, but PyTorch could not load there due to a Windows `fbgemm.dll` import failure, so MA-SAC training was run in `traffic_dqn` while keeping the benchmark code and seeds unchanged.

## Paper-method extraction
- Source paper: Towards Multi-agent Reinforcement Learning based Traffic Signal Control through Spatio-temporal Hypergraphs.
- Core method reported by the paper: MA-SAC with a spatio-temporal hypergraph-based critic for coordinating multiple traffic-signal agents over a network.

## Compatibility mapping
- Original paper setting: multiple intersections.
- Benchmark-fair adaptation here: one intersection with four movement-stage agents contributing action preferences for the same intersection controller.
- Original paper structure: spatio-temporal hypergraph critic.
- Benchmark-fair adaptation here: critic encodes the eight movement groups with fixed hyperedges and a temporal GRU before centralized Q estimation.
- Original paper execution: multi-agent traffic-network control.
- Benchmark-fair adaptation here: decentralized agent heads with a centralized critic, collapsed to one benchmark action among the same four actions used by our baseline setup.

## Risks / deviations
- This is not a paper-faithful multi-intersection reproduction because the benchmark has only one intersection.
- The hypergraph is constructed over movement groups inside the single intersection rather than over neighboring intersections.
- Results are suitable for the benchmark-fair comparison table, but they should be labeled as an adaptation rather than an exact reproduction of the original paper's experimental setting.
