from time import time
import torch as th
import torch.nn.functional as F
import torch.optim as optim

from alg.ppo.agent import Agent as PPOAgent
from .config import get_alg_args
from common.checkpoint import CheckpointSaver
from common.logger import Logger
from env.eval import Evaluator
from common.imports import *

class DAGGER:
    """
    DAgger for topology control (discrete actions) using a PPO actor checkpoint as expert.
    """

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CheckpointSaver):

        # Load algorithm args unless resuming
        if not ckpt.resumed:
            args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        assert args.action_type == "topology", "This DAgger implementation is for topology (discrete) only."

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        # Logger / evaluator (same pattern as PPO uses)
        logger = Logger(run_name, args)
        evaluator = Evaluator(args, logger, device)

        # Load PPO expert from checkpoint
        expert_run = th.load(args.expert_ckpt, map_location=device, weights_only=False)
        expert_args = expert_run["args"]
        expert = PPOAgent(envs, expert_args, continuous_actions=False).to(device)
        expert.actor.load_state_dict(expert_run["actor"])
        expert.actor.eval()

        # Student policy (reuse PPOAgent; we only train its actor)
        student = PPOAgent(envs, args, continuous_actions=False).to(device)
        student.actor.train()

        actor_optim = optim.Adam(student.actor.parameters(), lr=args.bc_lr, eps=1e-5)

        if ckpt.resumed:
            # resume student
            student.actor.load_state_dict(ckpt.loaded_run["actor"])
            actor_optim.load_state_dict(ckpt.loaded_run["actor_optim"])
            init_iter = ckpt.loaded_run.get("last_iter", 0) + 1
        else:
            init_iter = 0

        D_obs = []
        D_act = []
        dataset_size = 0

        # Reset envs
        obs, _ = envs.reset()
        global_step = 0

        # Helper: deterministic expert action = argmax logits.
        # We use argmax (greedy) rather than the stochastic policy so the expert labels are deterministic, as DAgger assumes.
        def expert_label(obs_t: th.Tensor) -> th.Tensor:
            logits = expert.actor(obs_t)
            return th.argmax(logits, dim=1)

        for it in range(init_iter, args.dagger_iters):
            # beta schedule (teacher forcing probability)
            beta = max(args.beta_final, args.beta_init * (args.beta_decay ** it))

            # rollout
            for t in range(args.rollout_steps):
                obs_t = th.tensor(obs, dtype=th.float32, device=device).view(args.n_envs, -1)

                with th.no_grad():
                    # student action distribution
                    student_logits = student.actor(obs_t)
                    student_action = th.argmax(student_logits, dim=1)

                    # expert label
                    a_star = expert_label(obs_t)

                    # mixture action for env interaction
                    use_expert = (th.rand(args.n_envs, device=device) < beta)
                    a_exec = th.where(use_expert, a_star, student_action)

                # store (s, a*)
                D_obs.append(obs_t.detach().cpu())
                D_act.append(a_star.detach().cpu())
                dataset_size += args.n_envs

                # cap dataset by number of stored transitions (FIFO), always
                # keeping at least one chunk
                while dataset_size > args.dataset_max and len(D_obs) > 1:
                    dataset_size -= D_obs[0].shape[0]
                    D_obs.pop(0)
                    D_act.pop(0)

                next_obs, rew, term, trunc, infos = envs.step(a_exec.cpu().numpy())
                obs = next_obs

                global_step += args.n_envs

            # Behaviour Cloning update on aggregated dataset
            student.actor.train()

            X = th.cat(D_obs, dim=0)  # [N, obs_dim]
            Y = th.cat(D_act, dim=0).long()  # [N]

            N = X.shape[0]

            for epoch in range(args.bc_epochs):
                idx = th.randperm(N)  # reshuffle each epoch
                for start in range(0, N, args.bc_batch_size):
                    batch_idx = idx[start:start + args.bc_batch_size]
                    xb = X[batch_idx].to(device)
                    yb = Y[batch_idx].to(device)

                    logits = student.actor(xb)
                    loss = F.cross_entropy(logits, yb)

                    actor_optim.zero_grad(set_to_none=True)
                    loss.backward()
                    actor_optim.step()

            # eval
            if (it % args.eval_every) == 0:
                student.actor.eval()
                evaluator.evaluate(global_step, student)

            # checkpoint
            if args.checkpoint and (it % args.checkpoint_every) == 0:
                ckpt.set_record(
                    args=args,
                    actor=student.actor,
                    global_step=global_step,
                    actor_optim=actor_optim,
                    last_iter=it,
                    dataset_size=dataset_size,
                )
                ckpt.save()

            if args.verbose:
                print(f"[DAgger] iter={it} beta={beta:.3f} dataset={dataset_size} global_step={global_step}")

        # final save
        if args.checkpoint:
            ckpt.set_record(
                args=args,
                actor=student.actor,
                global_step=global_step,
                actor_optim=actor_optim,
                last_iter=args.dagger_iters - 1,
                dataset_size=dataset_size,
            )
            ckpt.save()