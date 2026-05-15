# Pri-DDQN Benchmark Note

## Paper
- Pri-DDQN: learning adaptive traffic signal control strategy through a hybrid agent
- DOI: https://doi.org/10.1007/s40747-024-01651-5

## Extracted method components
- Double DQN backbone
- Priority-based dynamic experience replay
- Power-function exploration decay
- Asynchronous target network updates
- State and reward incorporated into loss / replay importance

## Benchmark-fair adaptation
- The original paper is single-intersection ATSC.
- Here it is implemented on two repository benchmarks:
  - original 2-action single-intersection setup
  - movement-based 4-stage 4-way setup
- Reward, horizon, seeds, and evaluation style were kept benchmark-consistent.
- Because the repository state vectors are compact rather than image-like DTSE tensors, the network uses a lightweight 1D-convolution feature extractor instead of a large image CNN.

## Environment note
- Requested environment `traffic-rl` was not usable for PyTorch training on this machine due to a Windows DLL import failure.
- Pri-DDQN training and evaluation were run with `traffic_dqn`, while keeping benchmark code and seeds unchanged.
