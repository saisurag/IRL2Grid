from time import time

from .agent import Agent
from .config import get_alg_args
from common.checkpoint import CheckpointSaver
from common.imports import *
from common.logger import Logger
from env.eval import Evaluator
from sklearn.tree import DecisionTreeClassifier, export_text

try:
    from pysr import PySRRegressor
    _PYSR_AVAILABLE = True
except ImportError:
    _PYSR_AVAILABLE = False


class SReinforce:
    """S-REINFORCE: PPO extended with a periodic Symbolic Regressor (SR) component.

    The SR is fitted periodically to imitate the NN actor's greedy action choices.
Its predictions are used to weight PPO advantages based on symbolic/NN action agreement.

    Reference: Dutta et al., "S-REINFORCE: A Neuro-Symbolic Policy Gradient Approach
    for Interpretable Reinforcement Learning", arXiv:2305.07367.
    """

    def __init__(self, envs: gym.Env, run_name: str, start_time: float,
                 args: Dict[str, Any], ckpt: CheckpointSaver):
        """Initialise S-REINFORCE.

        Args:
            envs (gym.Env): Vectorised training environments.
            run_name (str): Name of the current training run.
            start_time (float): Wall-clock time at training start.
            args (Dict[str, Any]): Command-line arguments.
            ckpt (CheckpointSaver): Checkpoint handler.
        """
        if not _PYSR_AVAILABLE:
            raise ImportError(
                "PySR is required for S-REINFORCE. Install it with: pip install pysr"
            )

        if not ckpt.resumed:
            alg_args = get_alg_args()

            for key, value in vars(alg_args).items():
                setattr(args, key, value)

        assert args.n_steps % args.n_envs == 0, \
            f"Invalid n_steps: {args.n_steps}. Must be a multiple of n_envs={args.n_envs}"

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        # ── Sizes ──────────────────────────────────────────────────────────────
        batch_size      = int(args.n_envs * args.n_steps)
        minibatch_size  = int(batch_size // args.n_minibatches)
        n_rollouts      = args.total_timesteps // batch_size
        init_rollout    = 1 if not ckpt.resumed else ckpt.loaded_run['last_rollout']

        continuous_actions = True if args.action_type == "redispatch" else False

        if continuous_actions:
            raise NotImplementedError(
                "Current S-REINFORCE implementation only supports discrete topology actions."
            )

        n_actions = envs.single_action_space.n

        # ── Agent ──────────────────────────────────────────────────────────────
        agent = Agent(envs, args, continuous_actions).to(device)
        if ckpt.resumed:
            agent.actor.load_state_dict(ckpt.loaded_run['actor'])
            agent.critic.load_state_dict(ckpt.loaded_run['critic'])

        actor_params = list(agent.actor.parameters())
        if continuous_actions:
            actor_params = actor_params + [agent.logstd]
        actor_optim  = optim.Adam(actor_params, lr=args.actor_lr, eps=1e-5)
        critic_optim = optim.Adam(agent.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        if ckpt.resumed:
            actor_optim.load_state_dict(ckpt.loaded_run['actor_optim'])
            critic_optim.load_state_dict(ckpt.loaded_run['critic_optim'])

        # ── Rollout buffers ───────────────────────────────────────────────────
        obs_shape  = envs.single_observation_space.shape
        act_shape  = envs.single_action_space.shape

        observations = th.zeros((args.n_steps, args.n_envs) + obs_shape).to(device)
        actions      = th.zeros((args.n_steps, args.n_envs) + act_shape).to(device)
        logprobs     = th.zeros((args.n_steps, args.n_envs)).to(device)
        rewards      = th.zeros((args.n_steps, args.n_envs)).to(device)
        dones        = th.zeros((args.n_steps, args.n_envs), dtype=th.int32).to(device)
        terminations = th.zeros((args.n_steps, args.n_envs), dtype=th.int32).to(device)
        values       = th.zeros((args.n_steps, args.n_envs)).to(device)

        # ── S-REINFORCE state ─────────────────────────────────────────────────
        sym_policy    = None          # PySRRegressor, None until first fit
        sr_states_buf = []            # obs  accumulated across rollouts
        sr_targets_buf = []           # NN greedy action labels accumulated
        # Feature-selection mask for SR (chosen once on first rollout by variance)
        sr_feature_mask: Optional[np.ndarray] = None

        # ── Logging / eval ────────────────────────────────────────────────────
        assert args.eval_freq % args.n_envs == 0, \
            f"Invalid eval_freq: {args.eval_freq}. Must be a multiple of n_envs={args.n_envs}"
        logger    = Logger(run_name, args) if args.track else None
        evaluator = Evaluator(args, logger, device)

        global_step = 0 if not ckpt.resumed else ckpt.loaded_run['global_step']
        next_obs, _ = envs.reset()
        next_obs    = th.tensor(next_obs).to(device)

        # ── Helper: select SR input features ──────────────────────────────────
        def _select_sr_features(obs_np: np.ndarray) -> np.ndarray:
            """Return the top-k highest-variance columns of obs_np."""
            nonlocal sr_feature_mask
            if sr_feature_mask is None:
                k = min(args.sr_max_obs_features, obs_np.shape[1])
                variances = obs_np.var(axis=0)
                sr_feature_mask = np.argsort(variances)[-k:]
                if args.verbose:
                    print(f"[S-REINFORCE] SR feature mask selected: {len(sr_feature_mask)} features")
            return obs_np[:, sr_feature_mask]

        # ── Helper: fit symbolic regressor ────────────────────────────────────
        def _fit_sr(states_np: np.ndarray, targets_np: np.ndarray):
            tree = DecisionTreeClassifier(
                max_depth=args.sr_maxsize,
                min_samples_leaf=20,
                random_state=args.seed,
            )
            tree.fit(states_np, targets_np.astype(np.int64))
            return tree
        
        def _symbolic_predict_actions(sym_pol,
                              b_obs_t: th.Tensor,
                              n_actions: int) -> th.Tensor:
            obs_np = b_obs_t.cpu().numpy()
            obs_feats = _select_sr_features(obs_np)

            pred_actions = sym_pol.predict(obs_feats).astype(np.int64)
            pred_actions = np.clip(pred_actions, 0, n_actions - 1)

            return th.tensor(pred_actions, dtype=th.long, device=device)

        # ══════════════════════════════════════════════════════════════════════
        # Training loop
        # ══════════════════════════════════════════════════════════════════════
        try:
            for iteration in range(init_rollout, n_rollouts + 1):

                # ── LR annealing ───────────────────────────────────────────────
                if args.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / n_rollouts
                    actor_optim.param_groups[0]['lr']  = frac * args.actor_lr
                    critic_optim.param_groups[0]['lr'] = frac * args.critic_lr

                # ── Rollout collection ─────────────────────────────────────────
                for step in range(0, args.n_steps):
                    global_step += args.n_envs
                    observations[step] = next_obs

                    with th.no_grad():
                        action, logprob, _ = agent.get_action(next_obs)
                        value              = agent.get_value(next_obs)
                        values[step]       = value.flatten()
                    actions[step]  = action
                    logprobs[step] = logprob

                    next_obs, reward, next_terminations, next_truncations, infos = \
                        envs.step(action.cpu().numpy())

                    rewards[step]      = th.tensor(reward).to(device).view(-1)
                    dones[step]        = th.tensor(
                        np.logical_or(next_terminations, next_truncations)).to(device)
                    terminations[step] = th.tensor(next_terminations).to(device)

                    real_next_obs = next_obs.copy()
                    for idx, done in enumerate(dones[step]):
                        if done:
                            real_next_obs[idx] = infos["final_observation"][idx]

                    next_obs      = th.tensor(next_obs).to(device)
                    real_next_obs = th.tensor(real_next_obs).to(device)

                    if global_step % args.eval_freq == 0:
                        evaluator.evaluate(global_step, agent)
                        if args.verbose:
                            print(f"SPS={int(global_step / (time() - start_time))}")

                # ── GAE bootstrap ──────────────────────────────────────────────
                with th.no_grad():
                    advantages  = th.zeros_like(rewards).to(device)
                    lastgaelam  = 0
                    for t in reversed(range(args.n_steps)):
                        if t == args.n_steps - 1:
                            nextvalues = agent.get_value(real_next_obs).reshape(1, -1)
                        else:
                            nextvalues = values[t + 1]
                        delta          = (rewards[t]
                                          + args.gamma * nextvalues * (1 - terminations[t])
                                          - values[t])
                        advantages[t]  = lastgaelam = (delta
                                                       + args.gamma * args.gae_lambda
                                                       * (1 - dones[t]) * lastgaelam)
                    returns = advantages + values

                # ── Flatten batch ──────────────────────────────────────────────
                b_obs        = observations.reshape((-1,) + obs_shape)
                b_logprobs   = logprobs.reshape(-1)
                b_actions    = actions.reshape((-1,) + act_shape)
                b_advantages = advantages.reshape(-1)
                b_returns    = returns.reshape(-1)
                b_values     = values.reshape(-1)

                # ── S-REINFORCE: accumulate data for SR ────────────────────────
                with th.no_grad():
                    nn_probs_flat = th.softmax(
                        agent.actor(b_obs), dim=-1
                    ).cpu().numpy()

                obs_flat_np = b_obs.cpu().numpy()
                sr_states_buf.append(obs_flat_np)

                sr_targets_buf.append(nn_probs_flat.argmax(axis=1))

                # ── S-REINFORCE: refit SR every sr_interval rollouts ───────────
                if iteration % args.sr_interval == 0:
                    all_states  = np.concatenate(sr_states_buf,  axis=0)
                    all_targets = np.concatenate(sr_targets_buf, axis=0)

                    sr_feats   = _select_sr_features(all_states)
                    sym_policy = _fit_sr(sr_feats, all_targets)

                    sr_states_buf.clear()
                    sr_targets_buf.clear()

                    if args.verbose:
                        train_acc = sym_policy.score(sr_feats, all_targets.astype(np.int64))
                        print(f"[S-REINFORCE] symbolic tree refit at rollout {iteration} "
                            f"| imitation_acc={train_acc:.3f}")
                        print(export_text(sym_policy, max_depth=3))

                # ── S-REINFORCE: IS weights (ones until first SR fit) ──────────
                if sym_policy is not None:
                    with th.no_grad():
                        symbolic_actions = _symbolic_predict_actions(sym_policy, b_obs, n_actions)

                    b_taken_actions = b_actions.reshape(-1).long()

                    # Weight samples higher when symbolic policy agrees with the taken action
                    b_is_weights = th.where(
                        symbolic_actions == b_taken_actions,
                        th.tensor(args.is_clip_high, device=device),
                        th.tensor(args.is_clip_low, device=device),
                    )
                    if args.verbose:
                        agreement = (symbolic_actions == b_taken_actions).float().mean().item()
                        print(f"[S-REINFORCE] symbolic-policy agreement={agreement:.3f}")
                else:
                    b_is_weights = th.ones(b_advantages.shape[0], device=device)

                # ── PPO update ─────────────────────────────────────────────────
                b_inds    = np.arange(batch_size)
                clipfracs = []

                for _ in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, batch_size, minibatch_size):
                        end      = start + minibatch_size
                        mb_inds  = b_inds[start:end]

                        _, newlogprob, entropy = agent.get_action(
                            b_obs[mb_inds], b_actions.long()[mb_inds]
                        )
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio    = logratio.exp()

                        with th.no_grad():
                            approx_kl  = ((ratio - 1) - logratio).mean()
                            clipfracs += [
                                ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                            ]

                        mb_advantages = b_advantages[mb_inds]
                        if args.norm_adv:
                            mb_advantages = (
                                (mb_advantages - mb_advantages.mean())
                                / (mb_advantages.std() + 1e-8)
                            )

                        # Apply IS weights from symbolic policy
                        mb_advantages = mb_advantages * b_is_weights[mb_inds]

                        # Policy loss (PPO-clip)
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * th.clamp(
                            ratio, 1 - args.clip_coef, 1 + args.clip_coef
                        )
                        pg_loss      = th.max(pg_loss1, pg_loss2).mean()
                        entropy_loss = entropy.mean()
                        pg_loss      = pg_loss - args.entropy_coef * entropy_loss

                        actor_optim.zero_grad()
                        pg_loss.backward()
                        nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                        actor_optim.step()

                        # Value loss
                        newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
                        if args.clip_vfloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped        = b_values[mb_inds] + th.clamp(
                                newvalue - b_values[mb_inds],
                                -args.clip_coef, args.clip_coef,
                            )
                            v_loss_clipped   = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss           = 0.5 * th.max(
                                v_loss_unclipped, v_loss_clipped
                            ).mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        v_loss = v_loss * args.vf_coef

                        critic_optim.zero_grad()
                        v_loss.backward()
                        nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                        critic_optim.step()

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break

                # ── Time-limit exit ────────────────────────────────────────────
                if (time() - start_time) / 60 >= args.time_limit:
                    break

        finally:
            ckpt.set_record(
                args, agent.actor, agent.critic, global_step,
                actor_optim, critic_optim,
                "" if not logger else logger.wb_path,
                iteration
            )
            ckpt.save()
            if logger:
                logger.close()
            envs.close()
