# Main RL Experiment: Movement-Based Controller

This package is the new primary RL benchmark for the 4-way intersection.

Decision actions:
- `0`: NS Straight, mapped to future SUMO phase index `0`
- `1`: NS Left / U-turn, mapped to future SUMO phase index `2`
- `2`: EW Straight, mapped to future SUMO phase index `4`
- `3`: EW Left / U-turn, mapped to future SUMO phase index `6`

Notes:
- The user request said "6 action spaces", but the supplied action table defines 4 RL decision actions and implied transition phases in between. This package implements the 4 decision actions directly.
- The experiment here is RL-only and does not run SUMO deployment.
- All artifacts are isolated under this folder.

Run:

```powershell
conda run -n traffic-rl python main_experiment_6action/train_classical.py
```
