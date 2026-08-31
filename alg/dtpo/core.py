import copy
from collections import deque
from time import time

from sklearn.tree import DecisionTreeRegressor

from .agent import Critic, RegressionTreePolicy
from .config import get_alg_args
from alg.distill import load_oracle_score_fn, distill
from common.action_reduction import ActionReducer
from common.checkpoint import CheckpointSaver
from common.imports import *
from common.logger import Logger
from env.eval import Evaluator


class DTPO:
    """
    Decision Tree Policy Optimization (Vos & Verwer, 2024, https://arxiv.org/abs/2408.11632).

    Optimizes a single hard regression-tree policy with policy gradients. Each iteration:
        (1) roll the (stochastic) tree policy out in the vectorized env
        (2) estimate GAE advantages with an NN critic
        (3) shift each visited state's action-probability target along the policy gradient
        (4) refit a freshmulti-output regression tree to those targets, keeping it only if it improves the DTPO objective
        (5) update the critic. 
    The returned policy is the best deterministic (argmax-leaf) tree by evaluated survival.
    """

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CheckpointSaver):
        if not ckpt.resumed:
            args = ap.Namespace(**vars(args), **vars(get_alg_args()))
        assert args.action_type == "topology", "DTPO is for discrete (topology) action spaces."

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")
        n_envs = args.n_envs
        n_actions = int(envs.single_action_space.n)
        obs_shape = envs.single_observation_space.shape
        steps_per_iter = max(1, args.dtpo_batch // n_envs)

        logger = Logger(run_name, args) if args.track else None
        evaluator = Evaluator(args, logger, device)

        critic = Critic(envs, args).to(device)
        critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)

        # optional hybrid warm-start: seed with a soft-distilled tree
        reducer, init_policy = None, None
        if getattr(args, "warmstart_oracle", ""):
            score_fn = load_oracle_score_fn(args.warmstart_oracle, envs, device, args.warmstart_oracle_type)
            ws_seeds = list(range(args.seed, args.seed + args.warmstart_seeds))
            if args.warmstart_curate:
                reducer = ActionReducer.from_oracle(
                    score_fn, evaluator.env, evaluator.max_steps, seeds=ws_seeds,
                    strategy=args.warmstart_curate_strategy, coverage=args.warmstart_coverage)
                reducer.log()
            init_policy, dinfo = distill(
                score_fn, evaluator.env, ws_seeds, n_actions=n_actions, device=device,
                mode="soft", temperature=args.warmstart_temperature, action_reducer=reducer,
                max_leaf_nodes=args.tree_max_leaf_nodes, max_depth=args.tree_max_depth,
                min_samples_leaf=args.tree_min_samples_leaf, steps_cap=args.warmstart_steps_cap,
                dagger_rounds=getattr(args, "warmstart_dagger_rounds", 0),
                dagger_steps_cap=getattr(args, "warmstart_dagger_steps_cap", None))
            print(f"[HYBRID] warm-start from {args.warmstart_oracle}: {dinfo}")

        # policy action dimension and curated->full env-action map
        n_pol = reducer.n_curated if reducer is not None else n_actions
        action_map = reducer.curated_to_full if reducer is not None else None

        def make_policy(tree):
            return RegressionTreePolicy(n_pol, device, tree=tree, action_map=action_map)

        if init_policy is not None:
            init_policy.action_map = action_map
            policy = init_policy
        else:
            policy = make_policy(None)

        best_policy, best_score = None, -np.inf
        candidates = []
        global_step, it = 0, 0
        rng = np.random.default_rng(args.seed)

        # survival-gated acceptance state (extension, --dtpo-survival-gate)
        gate_on = bool(getattr(args, "dtpo_survival_gate", False))
        gate_patience = int(getattr(args, "dtpo_gate_patience", 0))
        deployed_policy, deployed_surv, deployed_std = None, -np.inf, 0.0
        gate_rejects = 0

        # replay aggregation state (extension, --dtpo-replay-iters)
        replay_M = max(1, int(getattr(args, "dtpo_replay_iters", 1)))
        replay_clip = float(getattr(args, "dtpo_replay_ratio_clip", 2.0))
        replay_buf = deque(maxlen=replay_M)             # (obs, act, adv, behavior action-probs)

        # Deterministic, reproducible in-loop eval
        eval_total = int(getattr(args, "dtpo_eval_total", 0))
        eval_ids = evaluator.fixed_chronic_ids(eval_total) if eval_total > 0 else None
        if eval_ids is not None:
            print(f"[DTPO] deterministic eval on {len(eval_ids)} fixed chronics "
                  f"(of {evaluator.n_chronics()}): ids={eval_ids.tolist()}")

        tree_kwargs = dict(max_leaf_nodes=args.tree_max_leaf_nodes, max_depth=args.tree_max_depth,
                           min_samples_leaf=args.tree_min_samples_leaf, random_state=args.seed)

        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = np.asarray(next_obs, dtype=np.float32)

        try:
            for it in range(1, args.dtpo_iters + 1):
                obs_buf = np.zeros((steps_per_iter, n_envs) + obs_shape, dtype=np.float32)
                act_buf = np.zeros((steps_per_iter, n_envs), dtype=np.int64)
                rew_buf = np.zeros((steps_per_iter, n_envs), dtype=np.float32)
                term_buf = np.zeros((steps_per_iter, n_envs), dtype=np.float32)
                done_buf = np.zeros((steps_per_iter, n_envs), dtype=np.float32)
                val_buf = np.zeros((steps_per_iter, n_envs), dtype=np.float32)
                real_next_obs = next_obs.copy()

                # collect a batch of experience under the current tree policy
                for step in range(steps_per_iter):
                    obs_buf[step] = next_obs
                    with th.no_grad():
                        val_buf[step] = critic(th.as_tensor(next_obs, dtype=th.float32, device=device)).cpu().numpy()
                    actions = policy.sample(next_obs, rng)          # curated-space indices
                    act_buf[step] = actions
                    env_actions = actions if action_map is None else action_map[actions]
                    nobs, reward, term, trunc, infos = envs.step(env_actions)
                    rew_buf[step] = reward
                    done = np.logical_or(term, trunc)
                    term_buf[step] = term
                    done_buf[step] = done

                    real_next_obs = np.asarray(nobs, dtype=np.float32).copy()
                    if done.any():
                        for idx in np.where(done)[0]:
                            real_next_obs[idx] = infos["final_observation"][idx]
                    next_obs = np.asarray(nobs, dtype=np.float32)
                    global_step += n_envs

                # GAE(lambda) advantages with the NN critic
                with th.no_grad():
                    last_val = critic(th.as_tensor(real_next_obs, dtype=th.float32, device=device)).cpu().numpy()
                adv = np.zeros_like(rew_buf)
                lastgae = np.zeros(n_envs, dtype=np.float32)
                for t in reversed(range(steps_per_iter)):
                    nextval = last_val if t == steps_per_iter - 1 else val_buf[t + 1]
                    delta = rew_buf[t] + args.dtpo_gamma * nextval * (1 - term_buf[t]) - val_buf[t]
                    lastgae = delta + args.dtpo_gamma * args.dtpo_gae_lambda * (1 - done_buf[t]) * lastgae
                    adv[t] = lastgae
                ret = adv + val_buf

                b_obs = obs_buf.reshape((-1,) + obs_shape)
                b_act = act_buf.reshape(-1)
                b_adv = adv.reshape(-1).astype(np.float64)
                b_ret = ret.reshape(-1)
                if args.dtpo_norm_adv:
                    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                # policy-gradient probability targets (gradient of Eq. 4 wrt logits)
                # grad_l [ sigma(l)_a / pi_old * A ] = A * (onehot(a) - pi),  with l = log pi.
                # eta optionally annealed to 0 over training (reference behavior).
                eta_t = (args.dtpo_eta * (1.0 - (it - 1) / max(1, args.dtpo_iters))
                         if getattr(args, "dtpo_anneal_lr", False) else args.dtpo_eta)
                P = policy.proba(b_obs)                                # pi_old(.|s)
                old_p = P[np.arange(len(P)), b_act]
                onehot = np.zeros_like(P)
                onehot[np.arange(len(P)), b_act] = 1.0
                clip_eps = float(getattr(args, "dtpo_ppo_clip", 0.0))

                if clip_eps > 0.0:
                    def _ppo_loss(Pp):
                        r = Pp[np.arange(len(Pp)), b_act] / (old_p + 1e-12)
                        return float(np.minimum(r * b_adv,
                                                np.clip(r, 1 - clip_eps, 1 + clip_eps) * b_adv).sum())
                    loss_before = _ppo_loss(P)                         # ratio==1 -> sum(b_adv)
                    run_policy = policy
                    best_tree, best_loss = None, loss_before
                    for _u in range(max(1, int(getattr(args, "dtpo_policy_updates", 1)))):
                        Pr = run_policy.proba(b_obs)
                        r = Pr[np.arange(len(Pr)), b_act] / (old_p + 1e-12)
                        # PPO min: gradient is zeroed where the clipped branch is active
                        clipped = (((b_adv > 0) & (r > 1 + clip_eps)) |
                                   ((b_adv < 0) & (r < 1 - clip_eps)))
                        coef = np.where(clipped, 0.0, b_adv * r)       # per-sample scalar
                        logits = np.log(Pr + 1e-8) + eta_t * (coef[:, None] * (onehot - Pr))
                        logits -= logits.max(axis=1, keepdims=True)
                        Y = np.exp(logits); Y /= Y.sum(axis=1, keepdims=True)
                        cand = DecisionTreeRegressor(**tree_kwargs)
                        cand.fit(b_obs, Y)
                        cand_policy = make_policy(cand)
                        loss_after = _ppo_loss(cand_policy.proba(b_obs))
                        if loss_after > best_loss:
                            best_loss, best_tree = loss_after, cand
                        run_policy = cand_policy
                    improved = best_tree is not None
                    if improved:
                        policy = make_policy(best_tree)
                    J_new = best_loss
                else:
                    replay_buf.append((b_obs, b_act, b_adv, old_p))
                    if len(replay_buf) > 1:
                        r_obs = np.concatenate([b[0] for b in replay_buf])
                        r_act = np.concatenate([b[1] for b in replay_buf])
                        r_adv = np.concatenate([b[2] for b in replay_buf])
                        r_beh = np.concatenate([b[3] for b in replay_buf])
                        Pr = policy.proba(r_obs)
                        clip_hi = replay_clip
                    else:
                        r_obs, r_act, r_adv, r_beh, Pr = b_obs, b_act, b_adv, old_p, P
                        clip_hi = np.inf
                    r_onehot = np.zeros_like(Pr)
                    r_onehot[np.arange(len(Pr)), r_act] = 1.0

                    ratio = np.clip(Pr[np.arange(len(Pr)), r_act] / (r_beh + 1e-12), 0.0, clip_hi)
                    coef = r_adv * ratio
                    logits = np.log(Pr + 1e-8) + eta_t * (coef[:, None] * (r_onehot - Pr))
                    logits -= logits.max(axis=1, keepdims=True)
                    Y = np.exp(logits)
                    Y /= Y.sum(axis=1, keepdims=True)

                    refit_every = max(1, int(getattr(args, "dtpo_structure_refit_every", 1)))
                    do_full_refit = (policy.tree is None) or ((it - 1) % refit_every == 0)
                    if do_full_refit:
                        new_tree = DecisionTreeRegressor(**tree_kwargs)
                        new_tree.fit(r_obs, Y)
                    else:
                        new_tree = self._leaf_value_refit(policy.tree, r_obs, Y)
                    new_policy = make_policy(new_tree)
                    # keep only if it improves the DTPO objective J = E[ pi(a|s)/beta * A ]
                    J_old = float(np.mean(ratio * r_adv))
                    new_p = new_policy.action_proba(r_obs, r_act)
                    J_new = float(np.mean(np.clip(new_p / (r_beh + 1e-12), 0.0, clip_hi) * r_adv))
                    improved = J_new > J_old
                    if improved:
                        policy = new_policy

                # update the value critic
                self._update_critic(critic, critic_optim, b_obs, b_ret, args, device)

                # periodic deterministic evaluation + best-tree tracking
                if it % args.dtpo_eval_every == 0:
                    reuse = gate_on and (policy is deployed_policy)
                    if reuse:
                        surv, surv_std = deployed_surv, deployed_std
                    elif eval_ids is not None:
                        res = evaluator.evaluate_fixed(policy, eval_ids)
                        surv = float(res["survival"])
                        surv_std = float(res.get("survival_std", 0.0))
                    else:
                        res = evaluator.evaluate(global_step, policy, eval_ep=getattr(args, "dtpo_eval_ep", 3))
                        surv, surv_std = float(res["survival"]), 0.0
                    if not reuse and policy.tree is not None:
                        candidates.append((surv, global_step, it, copy.deepcopy(policy.tree)))
                        if surv > best_score:
                            best_score = surv
                            best_policy = make_policy(copy.deepcopy(policy.tree))

                    # survival gate
                    gate_tag, cand_tag = "", ""
                    if gate_on:
                        cand_tag = f" cand={surv*100:.2f}%"
                        if reuse:
                            gate_tag = " gate=hold"
                        elif deployed_policy is None or surv >= deployed_surv:
                            deployed_policy, deployed_surv, deployed_std = policy, surv, surv_std
                            gate_rejects = 0
                            gate_tag = " gate=accept"
                        elif gate_patience > 0 and gate_rejects >= gate_patience:
                            deployed_policy, deployed_surv, deployed_std = policy, surv, surv_std
                            gate_rejects = 0
                            gate_tag = " gate=accept(forced)"
                        else:
                            gate_rejects += 1
                            policy = deployed_policy                    # revert the rollout policy
                            gate_tag = f" gate=revert({gate_rejects})"
                        log_surv, log_std = deployed_surv, deployed_std
                    else:
                        log_surv, log_std = surv, surv_std

                    nl = policy.tree.get_n_leaves() if policy.tree is not None else 1
                    mode_tag = ""
                    if clip_eps <= 0.0:
                        mode_tag = f" mode={'full' if do_full_refit else 'leaf'}"
                    print(f"[DTPO] it={it}/{args.dtpo_iters} step={global_step} "
                          f"J_new={J_new:.4f} {'kept' if improved else 'revert'}{mode_tag}{gate_tag}{cand_tag} "
                          f"leaves={nl} survival={log_surv*100:.2f}% best={best_score*100:.2f}%"
                          f"{' std=%.1f%%' % (log_std*100) if eval_ids is not None else ''}")

                if (time() - start_time) / 60 >= args.time_limit:
                    break

        finally:
            # Store top-K candidate trees for final selection, as in-training best is unreliable, the real best chosen later  
            topk = max(1, int(getattr(args, "dtpo_rerank_topk", 5)))
            ranked = sorted(candidates, key=lambda c: c[0], reverse=True)[:topk]
            cand_records = [{"tree": t, "iter": cit, "step": gstep, "cheap_survival": float(s)}
                            for (s, gstep, cit, t) in ranked]
            if cand_records:
                fallback_tree = cand_records[0]["tree"]
                fallback_score = cand_records[0]["cheap_survival"]
                print(f"[DTPO] stored {len(cand_records)} candidate trees for final selection "
                      f"(cheap survivals: {['%.1f%%' % (c['cheap_survival']*100) for c in cand_records]}). "
                      f"Run dtpo_select.py --ckpt <ckpt> --episodes 100 to pick the honest best.")
            else:
                fallback_tree = None if best_policy is None else best_policy.tree
                fallback_score = best_score

            ckpt.set_record(args=args, tree=fallback_tree, global_step=global_step,
                            wb_run_name="" if not logger else logger.wb_path,
                            last_iter=it, best_score=fallback_score,
                            action_kept=(reducer.kept if reducer is not None else None),
                            candidates=cand_records)
            ckpt.save()
            if logger:
                logger.close()
            envs.close()

    @staticmethod
    def _leaf_value_refit(tree, X, Y):
        new_tree = copy.deepcopy(tree)
        leaf_ids = new_tree.apply(X)
        for lid in np.unique(leaf_ids):
            mask = leaf_ids == lid
            new_tree.tree_.value[lid, :, 0] = Y[mask].mean(axis=0)
        return new_tree

    @staticmethod
    def _update_critic(critic, opt, b_obs, b_ret, args, device):
        X = th.as_tensor(b_obs, dtype=th.float32, device=device)
        Y = th.as_tensor(b_ret, dtype=th.float32, device=device)
        n = len(X)
        idx = np.arange(n)
        for _ in range(args.critic_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, args.critic_batch):
                mb = idx[s:s + args.critic_batch]
                loss = 0.5 * ((critic(X[mb]) - Y[mb]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                opt.step()
