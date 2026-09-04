# Interpretable RL2Grid: Benchmarking Interpretable RL for Power Grid Operations

MSc Advanced Computing, Imperial College London
Supervisor: Dr. Francesco Leofante

This GitHub repository accompanies the MSc thesis on Explainable Reinforcement Larning for Power Grid Operations. It is built upon the RL2Grid benchmark, which can be found at the following link: https://github.com/emarche/RL2Grid

## Setup

Create the conda environment:

    conda env create -f conda_env.yml
    conda activate rl2grid

Then install the rest with pip, in this order:

    pip install grid2op==1.12.1
    pip install lightsim2grid==0.10.3
    pip install scikit-learn==1.8.0
    pip install matplotlib==3.10.8
    pip install numpy==1.26.4 gymnasium==0.29.1
    pip install pysr==1.5.10 [only if you need to test S-REINFORCE]

Unzip the action spaces before running any experiments. Note that everything is CPU-bound.

## Grid environments

The grids come directly from Grid2Op (https://grid2op.readthedocs.io/en/latest/), and will be downloaded automatically during your first run utilising them.

    bus5       rte_case5_example
    bus14      l2rpn_case14_sandbox
    bus36-M    l2rpn_wcci_2020
    bus118-M   l2rpn_neurips_2020_track2_small

By default grid2op downloads into ~/data_grid2op. To put it elsewhere, set "data_path" in ~/.grid2opconfig.json. Do not let a multi-process run do the first download, the workers race each other and corrupt the archive. Pull each dataset once single-process first:

    python -c "import grid2op; grid2op.make('l2rpn_case14_sandbox')"

## Repository layout

    alg/            algorithms. dtpo/ is the tree learner; viper/, dagger/,
                    s_reinforce/ and distill.py are the comparison methods.
                    dqn/, ppo/, sac/, td3/, lagr_ppo/ are the benchmark's own.
    common/         action-space curation by oracle advantage, evaluation
                    metrics, checkpointing, the observation normaliser
    env/            grid construction, the evaluation protocol, rewards
    checkpoints/    the PPO teachers every reported result distils from

    April-Runs/     tests run during the month of April
    May-Runs/       tests run during the month of May
    June-Runs/      tests run uring the month of July, focusing primarily on DTPO
    July-Runs/      tests run during the month of July, during which defects were observed and fixed
    Final-Results/  the final tests run during July and August, which form the basis of the thesis

It should be noted that not all tests have all been backtracked and added to this repository. Significant work was also spent on replicating MAVIPER (https://arxiv.org/abs/2205.12449) within MARL2Grid-TR (https://openreview.net/forum?id=mpAMH1OyMO), as well as initial tests during the months of April and May on post-hoc methods such as AGUA (https://dl.acm.org/doi/10.1145/3718958.3754341). These were not included due to dead code and lack of correlation with the final thesis. 


## Evaluating a checkpoint

Use `honest_eval_any.py`. It is the scoring protocol behind every reported figure of the interpretable policies.

    python honest_eval_any.py --ckpt <run>.tar --total 80

`eval_idle.py` measures the do-nothing baseline under the same flags a run used.


## Rendered policies and checkpoints

`Final-Results/visuals/` holds, the decision tree as Graphviz source and SVG in two orientations, plus the training curve.

Model checkpoints are not in this repository due to size concerns as they are compressed tar archives.

The teachers are the exception, in `checkpoints/`, because every
reported result distils from them:

    final_PPO_bus14_T_0_0__I__1775940444_3936.tar
    final_PPO_bus5_T_0_0__I__1776174124_47043_45000000.tar
    final_PPO_bus36-M_T_100_0__I__1784927195_14574.tar
    PPO_bus118-M_T_100_0_H___1785168371_15041.tar 

The remaining checkpoints are available here, in a folder mirroring this repository's structure:

    https://imperiallondon-my.sharepoint.com/:f:/g/personal/ssl125_ic_ac_uk/IgAryJdUAyF4Rqv3JtcbAtKvAduQvIQqQKM3VKeEy0Y11Kc?e=cr4P5t

## Provenance

Built on RL2Grid (https://github.com/emarche/RL2Grid) at commit `2812a6f`, cloned 4 March 2026, MIT licensed, copyright (c) 2025 Enrico Marchesini. The original LICENSE is retained.

Upstream ships dqn, lagr_ppo, ppo, sac and td3. Everything else is this project's work.

Commit history is chronological and records the work as it happened, including the retractions.