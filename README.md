# Interpretable RL2Grid: Benchmarking Interpretable RL for Power Grid Operations

MSc Advanced Computing, Imperial College London

Supervisor: Dr. Francesco Leofante

This GitHub repository accompanies the MSc thesis on *Explainable Reinforcement Learning for Power Grid Operations*. It is built upon the RL2Grid benchmark, which can be found at the following link: https://github.com/emarche/RL2Grid [1]. This repository contains the source code for the algorithms, procedures with regards to training, validation, testing, and evaluation, as well as the code to generate visualisation of these RL policies. Everything reported in the thesis is present in `Final-Results/hpc/experiments`, and can be reproduced from them.

## Contents

- [Setup](#setup)
- [Grid environmets](#grid-environments)
- [Repository layout](#repository-layout)
- [Checkpoints](#checkpoints)
- [Reproducibility](#reproducibility)
- [Evaluation protocol](#evaluation-protocol)
- [Provenance](#provenance)
- [Additional notes](#additional-notes)
- [Declaration](#declaration)

## Setup

All work within this framework is CPU-bound. No use of a GPU is required. Results were produced using Linux on CX3, the primary HPC facility at Imperial College London, using an Intel Icelake Xeon Platinum 8358 CPU node (64 cores). Evaluation, visualisation, and analysis of the results was performed on a local machine using WSL, with an AMD Ryzen 7500F 6-core CPU.

Create the conda environment:

    mkdir checkpoint                    # all newly trained .tar checkpoints will be saved here
    conda env create -f conda_env.yml
    conda activate rl2grid

Then install the rest with pip, in this order:

    pip install grid2op==1.12.1
    pip install lightsim2grid==0.10.3
    pip install scikit-learn==1.8.0
    pip install matplotlib==3.10.8
    pip install numpy==1.26.4 gymnasium==0.29.1
    pip install pysr==1.5.10                        # only to run S-REINFORCE

Miniforge3 was used on CX3. If creating the conda environment is not working as intended due to conda not being detected, then run the following command first, before following the remaining steps as aforementioned:

    eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"

To not have any further issues with library clashes, you can also run the following commands, which was done on CX3:

    pip install "stable_baselines3==2.3.2" wandb pandas tqdm numba
    pip install --force-reinstall --no-deps "numpy==1.26.4" "gymnasium==0.29.1"

Ensure that the action spaces is always unzipped before running any experiments, as a run will fail without their presence. The zip file `env/action_spaces.zip` has been provided, and can be unzipped using the `unzip` command.

## Grid environments

The grids come directly from Grid2Op (https://grid2op.readthedocs.io/en/latest/) [2], and will be downloaded automatically during your first run utilising them.

    bus5       rte_case5_example
    bus14      l2rpn_case14_sandbox
    bus36-M    l2rpn_wcci_2020
    bus118-M   l2rpn_neurips_2020_track2_small

By default grid2op downloads into ``~/data_grid2op`. To put it elsewhere, set `data_path` in `~/.grid2opconfig.json`. Do not let a multi-process run do the first download, as the workers race each other and this will corrupt the archive. Pull each dataset once in a single-process first, for:

    python -c "import grid2op; grid2op.make('l2rpn_case14_sandbox')"

## Repository layout

    alg/                algorithms. dtpo/ [6] is the tree learner; viper/ [7], dagger/ [8], s_reinforce/ [9]
                        and distill.py are the comparison methods. dqn/, ppo/, sac/, td3/, lagr_ppo/ are the 
                        original benchmark's own.
    common/             action-space curation by oracle advantage, evaluation metrics, 
                        checkpointing, the observation normaliser.
    env/                grid construction, the evaluation protocol, rewards.
    checkpoints/        the PPO teachers every reported result distils from.

    April-Runs/         tests run during the month of April (exploratory work).
    May-Runs/           tests run during the month of May (exploratory work).
    June-Runs/          tests run during the month of June (DTPO trials and development).
    July-Runs/          tests run during the month of July (DTPO improvement, evaluation fix).
    Miscellaneous/      outdated files for previous tests

    Final-Results/  the final tests run during July and August, which form the basis of the thesis.
        hpc/
            experiments/        .txt files for running all the experiments
            runs/               one directory for each run, with logs
            logs/               all log files generated on CX3
            results.csv         collected results table
            *.py                python files for training and testing
            *.sh                shell files for training
        analysis/       scripts for analysis of results.csv
        visuals/        rendered tree (portrait and landscape) and plotted training curves

    main.py             main file that is run for testing all algorithms
    honest_eval_any.py  evaluator for interpretable algorithms
    eval_idle.py        evaluate environments on action 0: do-nothing
    distill_only.py     one-shot CART distillation
    fidelity_eval.py    teacher-vs-student action agreement
    viz_dtpo_tree.py    visualisation of tree
    plot_dtpo_curve.py  plotting training curve for a run
    dtpo_select.py      selecting the candidate tree

Directories pertaining runs from April to July as well as the Miscellaneous folder are retained for record of the work. Thesis reports results from `Final-Results/` only. 

Not all tests have all been backtracked and added to this repository. Significant work was also spent on replicating MAVIPER [3] within MARL2Grid-TR [4], as well as initial tests during the months of April and May on post-hoc methods such as AGUA [5]. These were not included due to dead code and lack of correlation with the final thesis. 

## Checkpoints

Model checkpoints are not in this repository due to size concerns as they are compressed tar archives.

The teachers are the exception, in `checkpoints/`, because every reported result distils from them:

    final_PPO_bus14_T_0_0__I__1775940444_3936.tar
    final_PPO_bus5_T_0_0__I__1776174124_47043_45000000.tar
    final_PPO_bus36-M_T_100_0__I__1784927195_14574.tar
    PPO_bus118-M_T_100_0_H___1785168371_15041.tar 

The remaining checkpoints are available in either of the following links, mirroring this repository's structure:

    https://github.com/saisurag/IRL2Grid-checkpoints

    https://drive.google.com/file/d/1Pb0b8U6JRertQdG9igteh-RTTkOoG417/view?usp=drive_link

## Reproducibility

`Final-Results/hpc/experiments` contain the files that dictate the experimental runs. These are generated by `hpc/gen_experiments.py` which document the grid, hyperparameters, and flags for each policy configuration.

The checkpoints have been provided as usage of computational resources would be significant for retraining. For rescoring these checkpoint policies without retraining, the two repositories must be cloned side-by-side, and then the checkpoint must be copied into its respective run directory. An example has been provided below:

    git clone https://github.com/saisurag/IRL2Grid-checkpoints ../IRL2Grid-checkpoints

    RUN=bus14_dtpo-full_L16_s100
    mkdir -p Final-Results/hpc/runs/$RUN/checkpoint
    cp ../IRL2Grid-checkpoints/Final-Results/hpc/runs/$RUN/checkpoint/*.tar Final-Results/hpc/runs/$RUN/checkpoint/

    python honest_eval_any.py --ckpt Final-Results/hpc/runs/$RUN/checkpoint/DTPO_bus14_T_100_0__I__1785884352_23428.tar --total 80 --eval-pool held

Alternatively, you can point the checkpoint path to the other folder as follows:

    python honest_eval_any.py --ckpt ../IRL2Grid-checkpoints/Final-Results/hpc/runs/bus14_dtpo-full_L16_s100/checkpoint/*.tar --total 80  --eval-pool held

If there exists more than one `.tar` file, prefer the file with `final_`, and/or the latest file based on the UNIX timestamp. Note that for the bus5 grid, it will be `--total 10`.

If rescoring is required in bulk, or to compile the files into a single directory, the following instruction can copy all checkpoints in one command:

    cp -rn ../IRL2Grid-checkpoints/Final-Results/hpc/runs/. Final-Results/hpc/runs/

To retrain any policy, navigate to its respective directory, and then run `main.py`. For example:

    cd Final-Results/hpc/runs/bus14_dtpo-full_L16_s100
    python ../../../../main.py <args> > train.log 2>&1

The reported survival rate is the value of `RESULT` of `honest_eval.log`. For one-shot distillation, use `distill_only.py` instead of `main.py`. 

To reproduce the full study, `Final-Results/hpc/pbs_array.sh` is used as the job script to run all experiments. It takes `LIST` from the relevant `Final-Results/hpc/experiments/` file, `ROOT` (directory of the repository root must be provided), and records metrics. Any directory that contains `DONE` will be skipped. `Final-Results/hpc/submit_array.sh` is to be run to submit these runs to the node cluster.

## Evaluation protocol

Use `honest_eval_any.py`. It is the scoring protocol behind every reported figure of the interpretable policies.

    python honest_eval_any.py --ckpt <run>.tar --total 80

`eval_idle.py` measures the do-nothing baseline under the same flags a run used. `Final-Results/analysis/oracle_honest_eval.py` measures the PPO teacher oracle baseline. The run seeds are trained between 100-109. The evaluation seed '--eval-seed' is fixed for all at 12345. `--eval-pool held` scores on chronics withheld from training (`--chronic-holdout 4` holds back every fourth chronic, for example). The rest of the results can be collated for analysis as follows:

    cd Final-Results/hpc
    python collect_results.py

    cd ../analysis
    export RL2GRID_RESULTS=../hpc/results.csv
    export RL2GRID_LOGS=./logs
    
    python idle_per_seed.py <grid> <comma separated list of seeds> <held-out set> <chronic-holdout> >> logs/idle_per_seed.log
    python oracle_honest_eval.py <grid> <comma separated list of seeds> <burn-in> >> logs/oracle_honest_eval.log

    python paired_bus14.py
    python paired_other_buses.py

    python fidelity_winner.py <path to directory of folder of a run> <path to PPO teacher oracle> 10 12345 >> logs/fidelity_winner_bus14.log
    python fidelity_log_to_csv.py logs/fidelity_winner_bus14.log logs/fidelity_winner_bus14.csv
    # fidelity_eval.py scores the final training tree. fidelity_winner.py is used to score the tree used during testing.

    python frontier_out.py  ../visuals/frontier.png
    python make_visuals_out.py --runs-dir ../hpc/runs --outroot ../visuals

## Provenance

Built on RL2Grid (https://github.com/emarche/RL2Grid) [1] at commit `2812a6f`, cloned 4 March 2026, MIT licensed, copyright (c) 2025 Enrico Marchesini. The original LICENSE is retained.

Upstream ships dqn, lagr_ppo, ppo, sac and td3. Everything else is this project's work.

Commit history is chronological and records the work as it happened, including the retractions.

## Declaration

I acknowledge the use of Claude Code (Anthropic, https://claude.ai/code) as an assistive tool for programming, such as the debugging of code and scripts, and the extraction and visualisation of metrics from raw logs. Its assistance was also used for LaTeX formatting of tables and figures in the thesis report. All research designs are solely the original intellectual work of the author, and no (semantic) content generated by Large Language Models (LLMs) has been presented as the author's own work.

## References

[1] Marchesini E, Donnot B, Crozier C, Dytham I, Merz C, Schewe L, et al. RL2Grid: Benchmarking reinforcement learning in power grid operations. arXiv preprint arXiv:250323101. 2025.

[2] Marot A, Donnot B, Dulac-Arnold G, Kelly A, O’Sullivan A, Viebahn J, et al. Learning to run a power network challenge: a retrospective analysis. In: NeurIPS 2020 competition and demonstration track. PMLR; 2021.

[3] Milani S, Zhang Z, Topin N, Shi ZR, Kamhoua C, Papalexakis EE, et al. Maviper: Learning decision tree policies for interpretable multi-agent reinforcement learning. In: Joint European conference on machine learning and knowledge discovery in databases. Springer; 2022. p. 251-66. 

[4] Marchesini E, Boguslawski E, Leite A, Amato C, Dussartre M, Schoenauer M, et al. MARL2Grid-TR: A multi-agent RL benchmark in power grid operations. In: International Conference on Learning Representations. vol. 2026; 2026. p. 134590-607.

[5] Patel S, Han D, Narodytska N, Jyothi SA. Agua: A concept-based explainer for learning-enabled systems. In: Proceedings of the ACM SIGCOMM 2025 Conference 2025 Sep 8 (pp. 329-346).

[6] Vos D, Verwer S. Optimizing interpretable decision tree policies for reinforcement learning. arXiv preprint arXiv:2408.11632. 2024 Aug 21.

[7] Bastani O, Pu Y, Solar-Lezama A. Verifiable reinforcement learning via policy extraction. Advances in neural information processing systems. 2018;31.

[8] Ross S, Gordon G, Bagnell D. A reduction of imitation learning and structured prediction to no-regret online learning. In: Proceedings of the fourteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings; 2011. p. 627-35.

[9] Dutta R, Wang Q, Singh A, Kumarjiguda D, Xiaoli L, Jayavelu S. S-reinforce: A neuro-symbolic policy gradient approach for interpretable reinforcement learning. arXiv preprint arXiv:230507367. 2023. 
