# Final RL Methodology

## Title
Hybrid-Enhanced Pri-Hyper DDQN for Movement-Based Adaptive Traffic Signal Control

## 1. Problem Setting

This project studies adaptive traffic signal control for a single isolated four-way intersection using the repository's movement-based 4-action benchmark in [main_experiment_6action/environments/traffic_env.py](/C:/Users/Ashritha/Downloads/final_rl/traffic_dqn_fixed/traffic_dqn/main_experiment_6action/environments/traffic_env.py).

The controller chooses one of four protected stages at each decision step:

- `0`: NS Straight
- `1`: NS Left / U-turn
- `2`: EW Straight
- `3`: EW Left / U-turn

The environment models eight movement groups:

- `n_through`
- `n_turn`
- `s_through`
- `s_turn`
- `e_through`
- `e_turn`
- `w_through`
- `w_turn`

Each decision interval is `10` seconds, with a `3` second yellow loss when the phase changes. The episode horizon is `3600` simulated seconds.

The reward used by the benchmark environment is:

```text
R =
  -0.45 * queue_pressure
  -0.30 * wait_penalty
  +0.20 * throughput_bonus
  -0.05 * fairness_penalty
  -0.05 * switch_penalty
```

This reward makes the task structurally multi-objective:

- minimize queue length
- minimize waiting time
- maximize throughput
- maintain fairness across movements
- avoid unnecessary switching

## 2. Motivation for a New Method

The repository already contains five comparison families:

- `TD(0)`
- `SARSA`
- `Q-Learning`
- `Pri-DDQN`
- `MA-SAC`

The classical tabular methods underperform badly on the 4-action benchmark because they rely on coarse state discretization and cannot generalize well across movement interactions.

The two deep baselines are much stronger, but they optimize different parts of the problem:

- `Pri-DDQN` is strong at reward-oriented value learning through double Q-learning, prioritized replay, and stable off-policy optimization.
- `MA-SAC` is strong at structured state modeling through temporal context and movement-level interaction encoding.

The new Hybrid-Enhanced method was designed to combine the strongest useful parts of both papers while remaining benchmark-fair on the exact same simulator, reward function, seeds, horizon, and action space.

## 3. Core Idea

The Hybrid-Enhanced method is:

```text
Prioritized Double Dueling Q-Learning
+ Temporal Hypergraph State Encoding
+ Pressure-Aware Action Prior
+ Conservative Phase Switching
```

In implementation terms, it is a hybrid between:

- the Pri-DDQN style training logic
- the MA-SAC style spatio-temporal movement representation
- an additional traffic-engineering prior layer not present in the existing baselines

This is why the method is named:

```text
Hybrid-Enhanced Pri-Hyper DDQN
```

## 4. Uniqueness of the Hybrid-Enhanced Method

This method is not just another DQN and not just a reduced SAC variant. Its uniqueness comes from combining three distinct design layers:

### 4.1 Algorithmic Uniqueness

It keeps a value-based control backbone rather than switching to actor-critic. That matters because the environment has:

- small discrete action space
- short action semantics
- explicit switch penalties
- relatively low-dimensional observation structure

For this setting, value-based control is more direct and easier to stabilize than entropy-regularized policy optimization.

### 4.2 Representation Uniqueness

Instead of feeding only a flat raw vector into the Q-network, Hybrid-Enhanced encodes the intersection as:

- movement-level nodes
- temporal state sequences
- hypergraph-style interaction structure

This is inspired by MA-SAC, but here it is used inside a Double-DQN style value estimator rather than a centralized SAC critic.

### 4.3 Decision-Layer Uniqueness

On top of learned Q-values, the method injects a pressure-aware prior that biases decisions toward:

- currently congested served movements
- heavily delayed served movements
- turn stages when turn backlogs are being starved
- smoother switching when the advantage to switch is small

This prior is not a fixed controller replacing RL. It is a structured inductive bias layered onto the RL policy.

### 4.4 Benchmark Uniqueness

Among all methods tested in this repository, Hybrid-Enhanced is the only one that simultaneously:

- uses deep value learning
- uses prioritized replay
- uses double-Q target selection
- uses a dueling head
- uses temporal movement-history input
- uses hypergraph-style relational encoding
- uses explicit traffic-pressure priors during action selection
- uses conservative switching thresholds at deployment

That combination is unique in this codebase.

## 5. System Architecture

The implementation is in [hybrid_enhanced_benchmark.py](/C:/Users/Ashritha/Downloads/final_rl/traffic_dqn_fixed/traffic_dqn/hybrid_enhanced_benchmark.py).

The architecture can be understood as six modules.

### 5.1 Environment Interface Layer

Input source:

- 4-action movement environment from `main_experiment_6action`

At each decision step, the environment provides a raw state vector of length `17`:

- `8` normalized queue values
- `8` normalized wait values
- `1` normalized current action / phase indicator

### 5.2 Temporal State Builder

The controller does not act on a single raw state only. It constructs a temporal window:

- temporal window size: `4`

This produces a state tensor:

```text
[time_steps, features] = [4, 17]
```

This preserves short-term dynamics:

- queue buildup trend
- persistence of waiting time
- recent phase context
- switching consequences across adjacent decisions

### 5.3 Hypergraph Temporal Encoder

The encoder treats the first `16` state features as `8` movement nodes, each with two attributes:

- queue
- wait

So for each time step:

```text
8 nodes x 2 features
```

The encoder pipeline is:

1. node projection
2. hypergraph message passing
3. temporal GRU aggregation
4. global embedding extraction

#### Node Projection

Each movement node feature pair is projected into a learnable embedding space:

- embedding dimension: `48`

#### Hypergraph Interaction Layer

A fixed incidence matrix defines structured relations among movements. This captures:

- directional couplings
- through-turn relationships
- axis-level interactions
- movement grouping patterns

This step is important because isolated movement queues are not independent. A good signal decision depends on coordinated interactions between:

- straight and turn demand
- north-south vs east-west axis pressure
- movement competition under shared signal time

#### Temporal Aggregation

After relational mixing, each movement's short history is processed with a GRU. This allows the network to detect:

- persistent growth
- recent release after service
- oscillatory switching effects
- emerging starvation

#### Global Embedding

The node embeddings are averaged into one global traffic-context embedding for downstream Q-value estimation.

### 5.4 Pri-DDQN Style Value Network

After encoding, the model uses a dueling value architecture:

- shared MLP trunk
- scalar value head
- 4-action advantage head

Final Q-values are computed as:

```text
Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
```

This improves stability when the state value matters separately from action-specific deviation.

### 5.5 Pressure-Aware Prior Layer

In parallel with the learned network, a prior score vector is computed from the latest state only.

For each action, the prior includes:

- served queue mass
- served waiting mass
- turn-stage congestion bonus for left/U-turn stages
- stay bonus for the current action

The prior score is normalized and blended with Q-values:

```text
blended_action_score = Q(s,a) + prior_weight * prior(a)
```

This is one of the key methodological contributions. It injects traffic-domain structure into deployment-time action selection without replacing learned Q-values.

### 5.6 Conservative Switching Layer

Even if another action has the largest blended score, the controller does not switch immediately unless the improvement over the current phase exceeds a threshold:

```text
if score(best_action) - score(current_action) < switch_margin:
    keep current action
```

This stabilizes control by reducing:

- jitter
- excessive yellow losses
- reward degradation from unnecessary switching

This layer directly addresses a weakness that emerged in early hybrid experiments: traffic-efficiency metrics improved, but reward lagged because of over-switching.

## 6. Training Methodology

### 6.1 Learning Paradigm

The method is off-policy deep reinforcement learning using Double DQN.

Two networks are maintained:

- `online` network
- `target` network

Action selection for target computation uses the online network, while target evaluation uses the target network. This reduces overestimation bias.

### 6.2 Replay Strategy

Experience replay is prioritized rather than uniform.

Each transition priority uses:

- TD error
- reward magnitude
- state magnitude

Priority formula:

```text
priority =
  |TD error|
  + reward_scale * |reward|
  + state_scale * mean(|state|)
  + epsilon
```

This is inherited from the Pri-DDQN spirit: important transitions are replayed more often.

### 6.3 Importance Weighting

During replay sampling, importance weights correct the bias introduced by prioritized replay. Beta is annealed during training from:

- `0.4` to `1.0`

### 6.4 Loss Function

The update objective is a weighted smooth L1 loss:

```text
loss =
  importance_weight
  * reward_weight
  * (
      smooth_l1(Q, target)
      + state_scale * state_penalty
    )
```

where:

- `reward_weight = 1 + reward_scale * |reward|`
- `state_penalty = mean(|state_seq|)`

This again follows Pri-DDQN's idea of making replay and loss sensitive to more than plain TD error alone.

### 6.5 Target Updates

The target network is updated softly:

```text
target <- (1 - tau) * target + tau * online
```

This avoids unstable hard jumps in target estimates.

### 6.6 Exploration Schedule

Exploration uses power-decay epsilon, adapted from Pri-DDQN:

- `eps_start = 1.0`
- `eps_end = 0.03`

This keeps stronger exploration early and decays more smoothly than naive linear decay.

### 6.7 Hyperparameters Used

The final hybrid configuration used:

- episodes: `360`
- temporal window: `4`
- gamma: `0.97`
- learning rate: `4e-4`
- batch size: `64`
- replay capacity: `60,000`
- warmup steps: `800`
- target update interval: `4`
- soft update tau: `0.04`
- embedding dimension: `48`
- hidden dimension: `96`
- priority alpha: `0.7`
- prior weight: `0.16`
- switch margin: `0.07`
- stay bonus: `0.10`

## 7. End-to-End Data Flow

The complete decision cycle is:

1. environment emits raw 17-dimensional state
2. state is appended to temporal history
3. temporal history is converted to a `[4, 17]` tensor
4. first 16 features are reshaped into `8` movement nodes with `2` features each
5. hypergraph message passing mixes movement relations
6. GRU summarizes short-term temporal dynamics
7. global embedding is formed
8. dueling Q-network predicts action values
9. pressure-aware prior scores are computed from the latest state
10. Q-values and prior scores are blended
11. conservative switching logic optionally suppresses weak switches
12. final action is sent to the environment

This architecture is more layered than the other methods:

- classical methods stop at state discretization and table lookup
- Pri-DDQN stops at flat-state deep Q estimation
- MA-SAC uses relational temporal encoding but via actor-critic policy optimization
- Hybrid-Enhanced combines relational encoding, prioritized value learning, and traffic-domain decision priors in one pipeline

## 8. Why It Is Better Than the Classical Methods

### 8.1 Compared with TD(0), SARSA, and Q-Learning

The classical methods fail mainly because:

- state space is too large for coarse tables
- movement interactions are not modeled well
- temporal dependence is weakly represented
- generalization is poor across unseen queue/wait combinations

Hybrid-Enhanced solves those issues by:

- continuous state representation
- deep function approximation
- relational movement modeling
- temporal sequence encoding
- replay-based sample reuse
- more stable deep target estimation

## 9. Why It Is Different from Pri-DDQN

Pri-DDQN in this repository is already strong because it has:

- Double DQN
- prioritized replay
- power-function epsilon decay
- reward/state-aware replay weighting

But it still uses a relatively flat raw-state processing pipeline.

Hybrid-Enhanced extends Pri-DDQN in three major ways:

### 9.1 Temporal Context

Pri-DDQN uses the current state only.

Hybrid-Enhanced uses the recent state sequence, which helps detect:

- trend
- backlog persistence
- delayed response to switching

### 9.2 Hypergraph Movement Representation

Pri-DDQN treats the state mostly as a compact feature vector.

Hybrid-Enhanced explicitly models movement groups and their structured relations before producing Q-values.

### 9.3 Pressure-Aware Deployment Bias

Pri-DDQN is fully data-driven at action selection time.

Hybrid-Enhanced adds a traffic-engineering inductive bias, which improved:

- queue
- wait
- delay

relative to Pri-DDQN on this benchmark.

## 10. Why It Is Different from MA-SAC

MA-SAC contributes the strongest representation idea in the repository:

- temporal modeling
- structured movement interactions

But on this benchmark, SAC-style control is not obviously ideal because:

- the action space is very small and discrete
- switch penalties create a sharp control tradeoff
- stable discrete value ranking matters more than stochastic policy entropy

Hybrid-Enhanced borrows the state-modeling advantages of MA-SAC while keeping the more suitable value-learning backbone of Pri-DDQN.

So compared with MA-SAC, Hybrid-Enhanced is:

- less dependent on entropy-driven policy exploration
- more directly aligned with discrete action ranking
- more compatible with replay prioritization
- easier to blend with traffic-specific priors

## 11. Benchmark Results

The final head-to-head 4-action results from [outputs/hybrid_enhanced/comparison.md](/C:/Users/Ashritha/Downloads/final_rl/traffic_dqn_fixed/traffic_dqn/outputs/hybrid_enhanced/comparison.md) are:

| Method | Reward | Std | Avg Queue | Avg Wait | Throughput | Delay | Phase Changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| TD(0) | -346.245 | 1.313 | 19.001 | 1359.515 | 717.900 | 547230.333 | 0.000 |
| SARSA | -87.240 | 71.912 | 3.301 | 103.922 | 1633.600 | 95076.000 | 217.833 |
| Q-Learning | -216.207 | 73.829 | 8.223 | 358.278 | 1385.733 | 236827.667 | 135.867 |
| Pri-DDQN | -19.586 | 0.616 | 0.973 | 21.294 | 1717.033 | 28008.000 | 244.000 |
| MA-SAC | -25.893 | 1.252 | 1.079 | 21.602 | 1722.333 | 31077.333 | 330.067 |
| Hybrid-Enhanced | -20.761 | 0.446 | 0.863 | 19.146 | 1714.067 | 24864.333 | 306.833 |

## 12. Result Interpretation

### 12.1 Best Overall Reward

On the benchmark's official reward:

- `Pri-DDQN` is best with `-19.586`
- `Hybrid-Enhanced` is second with `-20.761`

### 12.2 Best Traffic-Efficiency Controller

On physical traffic metrics, Hybrid-Enhanced is strongest:

- lowest average queue: `0.863`
- lowest average wait: `19.146`
- lowest total delay: `24864.333`

This is a meaningful result. It shows that the new method improved real traffic-serving behavior even though the benchmark reward still slightly favors Pri-DDQN.

### 12.3 Why Reward Did Not Fully Surpass Pri-DDQN

The main reason is phase switching.

Hybrid-Enhanced produced:

- better queue and waiting outcomes
- but more phase changes than Pri-DDQN

Since the environment penalizes switching, some operational gains are offset in the scalar reward. This is why the method beats Pri-DDQN on traffic-state metrics but not yet on final reward.

### 12.4 Practical Takeaway

If the goal is:

- strict benchmark reward maximization: Pri-DDQN is still best
- better traffic flow quality: Hybrid-Enhanced is better

## 13. Strengths of Hybrid-Enhanced

The new method's strengths are:

- combines the strongest deep RL components from both papers
- models movement relations explicitly
- uses short-term temporal memory
- keeps value-based action ranking for a discrete 4-action problem
- improves queue, wait, and delay beyond Pri-DDQN
- strongly outperforms all classical methods
- beats MA-SAC on all major metrics except throughput

## 14. Limitations

The current Hybrid-Enhanced version still has some limitations:

- reward is still slightly below Pri-DDQN
- switch suppression is heuristic rather than fully learned
- hypergraph structure is fixed manually
- no explicit phase-age feature is passed into the encoder beyond current phase
- the pressure prior is hand-designed and not meta-learned

## 15. Future Improvement Directions

The most promising next extensions are:

- add explicit phase-age and minimum-green features into the state
- learn the switching threshold adaptively
- add action-repeat duration prediction instead of one-step discrete switching only
- learn the prior blending weight instead of fixing it
- extend to a short model-predictive action evaluation head
- use multi-objective reward balancing or constrained RL to reduce switch penalties while preserving queue and wait gains

## 16. Final Summary

Hybrid-Enhanced is a new robust RL controller designed specifically for this repository's 4-action intersection benchmark. It is unique because it combines:

- Pri-DDQN style prioritized double dueling Q-learning
- MA-SAC style temporal hypergraph movement encoding
- a pressure-aware traffic prior
- conservative switching logic

Among all algorithms tested in this codebase, it is the most structurally rich and the most traffic-aware. It does not yet achieve the best scalar benchmark reward, but it is the strongest method on the most meaningful traffic-efficiency metrics:

- best average queue
- best average waiting time
- best total delay

That makes Hybrid-Enhanced the most operationally robust controller produced in this project so far.
