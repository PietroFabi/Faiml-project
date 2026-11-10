# Sim-to-Real Transfer with Reinforcement Learning

Training a simulated Franka Panda robot arm to push a block to a goal, and studying how well the learned policy survives a change in physics. A policy is trained in a **source** simulation (block mass 1 kg) and tested in a **target** simulation (block mass 5 kg), a stand-in for the gap between simulation and the real world. **Domain randomization** of the block's mass is then used to make the policy more robust.

Course project for *FAIML* (01VSDWS), Politecnico di Torino, academic year 2025/2026.

## What was done

All experiments use the `PandaPush-v3` environment (dense reward) with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

1. **Hyperparameter tuning** of PPO and SAC on the source domain (three configurations each).
2. **PPO vs SAC**: both trained on source, tested on source and target.
3. **Lower and upper bounds**: SAC trained on source and tested on target (lower bound), against SAC trained directly on target (upper bound).
4. **Uniform Domain Randomization (UDR)**: the block's mass is resampled uniformly from a range at every episode. Three ranges are compared.
5. **Automatic Domain Randomization (ADR)**: the mass range starts narrow around the nominal mass and widens or shrinks automatically depending on the agent's success rate at the range boundaries.

Supporting pieces written for the project:

- `rand_wrapper.py`: a Gymnasium wrapper that implements both UDR and ADR by changing the block's mass in PyBullet at every reset.
- `custom_callback.py`: an evaluation callback that keeps the observation-normalisation statistics in sync between training and evaluation environments, and saves them together with the best model.
- Training runs on 8 parallel environments (`SubprocVecEnv`) with observation and reward normalisation (`VecNormalize`).

## Results

Success rate and mean return over evaluation episodes with fixed seeds. Mean return is the negative distance to the goal accumulated over an episode, so closer to zero is better.

### Hyperparameter tuning (source-trained, 100 evaluation episodes)

| Model | Configuration | Success on source | Success on target |
| --- | --- | --- | --- |
| PPO A | SB3 defaults | 44% | 39% |
| PPO B | batch 256, entropy coef. 0.01 | 43% | 35% |
| **PPO C** | lr 1e-4, 4096 steps, batch 512, 20 epochs, entropy coef. 0.005, net [256, 256] | **81%** | **77%** |
| SAC A | SB3 defaults | 19% | 19% |
| SAC B | gamma 0.98, tau 0.01 | 19% | 19% |
| **SAC C** | net [256, 256, 256], one gradient step per parallel environment | **100%** | **99%** |

PPO models were trained for 10M steps, SAC models for 500k steps.

### Final experiments (50 evaluation episodes)

| Training setup | Source: success / mean return | Target: success / mean return |
| --- | --- | --- |
| PPO on source | 80% / -1.507 | 82% / -1.394 |
| SAC on source | 100% / -0.387 | 100% / -0.535 |
| SAC on source (second run, lower bound) | 100% / -0.377 | 100% / -0.494 |
| SAC on target (upper bound) | not evaluated | 100% / -0.417 |
| SAC + UDR, mass range 0.5 to 2.0 kg | 100% / -0.392 | 100% / -0.464 |
| SAC + ADR | 100% / -0.401 | 100% / -0.592 |

UDR range comparison (100 evaluation episodes): 0.5 to 2.0 kg gave 100% / 100% success (source / target), 0.5 to 6.0 kg gave 98% / 99%, and 0.1 to 10.0 kg gave 100% / 100%.

### Takeaways

- SAC solves the task with twenty times fewer environment steps than PPO, and far more reliably.
- The mass shift is mild for SAC: success stays at 100% on the target, so the sim-to-real gap shows up only in the mean return (-0.494 to -0.535 for source-trained SAC, against -0.417 when training directly on target).
- UDR recovers part of that gap (-0.464 on target).
- ADR did not help in this run (-0.592 on target, worse than no randomization).
- Every configuration was trained with a single seed, so small differences in mean return should be read with caution.

Full training and evaluation output is in the `part2/log_*.txt` files.

## Repository layout

```
├── requirements.txt
├── part1/                      Hopper starter template provided by the course
│   ├── agent.py
│   ├── train.py
│   ├── test_random_policy.py
│   └── colab_template/
└── part2/                      Panda push experiments (the work described above)
    ├── train_sb3.py            Final experiments; set CONFIG = 1, 2 or 3 at the top of the file
    ├── eval_sb3.py             Evaluate a saved model on source or target
    ├── rand_wrapper.py         UDR / ADR mass randomization wrapper
    ├── custom_callback.py      Evaluation callback that also saves normalisation statistics
    ├── HpT_PPO.py              PPO hyperparameter tuning
    ├── HpT_SAC.py              SAC hyperparameter tuning
    ├── HpT_SAC_UDR.py          UDR mass-range comparison
    ├── prova_SAC_b_*.py        Side experiments (different seed, different learning rate)
    ├── test_random_policy.py   Random policy on PandaPush, with rendering
    ├── log_*.txt               Console output of every run
    ├── best_models_PPO/        Saved models from PPO tuning
    ├── best_models_SAC/        Saved models from SAC tuning
    ├── best_models_UDR/        Saved models from the UDR range comparison
    ├── final_models_task4/     PPO vs SAC
    ├── final_models_task5/     SAC on source vs SAC on target
    ├── final_models_task6/     SAC with UDR vs SAC with ADR
    └── panda-gym/              Modified copy of panda-gym (see Credits)
```

Each model folder contains `best_model.zip` (the Stable-Baselines3 model) and `vec_normalize.pkl` (the normalisation statistics it needs).

## Setup

Python 3.11 was used.

```bash
pip install -r requirements.txt
cd part2/panda-gym
pip install -e .
```

## Usage

Run everything from the `part2` folder.

Train (pick the experiment with the `CONFIG` variable at the top of `train_sb3.py`: 1 = PPO vs SAC, 2 = source vs target, 3 = UDR vs ADR):

```bash
python train_sb3.py --timesteps 500000 --num-cpus 8
```

Evaluate a saved model on the target domain:

```bash
python eval_sb3.py \
  --model-path final_models_task6/SAC_UDR_Source/best_model.zip \
  --stats-path final_models_task6/SAC_UDR_Source/vec_normalize.pkl \
  --algo sac --env-type target
```

`eval_sb3.py` opens a rendering window by default.

Reproduce the hyperparameter tuning:

```bash
python HpT_PPO.py
python HpT_SAC.py
python HpT_SAC_UDR.py --sampling-strategy udr
```

The saved models and `.pkl` files use Python's pickle format, as is standard for Stable-Baselines3. Only load model files from sources you trust.

## Credits

- The project skeleton (`part1`, and the initial versions of the `part2` scripts) is the starting code provided by the course staff.
- `part2/panda-gym` is a copy of [panda-gym](https://github.com/qgallouedec/panda-gym) by Quentin Gallouédec et al., released under the MIT license (see `part2/panda-gym/LICENSE`). The push task was modified for the course to add the `source` / `target` domains with different block masses.
