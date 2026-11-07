#!/usr/bin/env python3
"""
BME1322 Assignment 3 - experiment entry point
=============================================

Reproduces every figure and table used in ``report.md``.

Part A (this file's current scope)
----------------------------------
* collect the fixed 2000-episode uniform-random dataset (seed 2026);
* estimate P_hat / R_hat with ``ModelBasedAgent``;
* run synchronous value iteration and extract the greedy policy;
* evaluate the policy on 500 fresh simulator episodes (undiscounted return,
  fraction of time in B3, rate of termination by entering B1);
* repeat with the first 500 / 1000 / 2000 episodes (data-budget ablation);
* save figures under ``assignment3/figures/`` and raw numbers under
  ``assignment3/results/``.

Usage
-----
    python assignment3/main.py --part a
    python assignment3/main.py --part all        # after Part B is added

The script must be run from the directory that contains ``assignment3/``
(the parent of this file) so that ``glucose_env.py`` can be imported, or
from anywhere: the parent directory is added to ``sys.path`` automatically.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
for p in (str(PARENT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from glucose_env import (  # noqa: E402
    ACTION_NAMES,
    GlucoseEnv,
    STATE_NAMES,
    collect_random_episodes,
)
from model_based import ModelBasedAgent  # noqa: E402
from q_learning import QLearningAgent  # noqa: E402

# --------------------------------------------------------------------------
# Global experiment settings (fixed for reproducibility)
# --------------------------------------------------------------------------
# The discount factor and episode horizon are specified by the handout.
# They are written here explicitly instead of importing environment
# internals, so the experiment code only uses the public simulator API.
GAMMA = 0.9
MAX_STEPS = 200
DATA_SEED = 2026
N_EPISODES = 2000
EVAL_EPISODES = 500
EVAL_SEED_BASE = 10_000
RANDOM_ACTION_SEED = 777
ABLATION_SIZES = (500, 1000, 2000)
SPARSE_THRESHOLD = 50

# Part B settings
ALPHA = 0.5
EPSILON = 0.3
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
ONLINE_EPISODES = 1000
ONLINE_SEEDS = (0, 1, 2, 3, 4)
ONLINE_WINDOW = 50
ONLINE_TRAIN_SEED_BASE = 1000
ONLINE_EVAL_EPISODES = 500
ONLINE_EVAL_SEED_BASE = 20_000
NONTERMINAL_STATES = [s for s in STATE_NAMES if s != "B1"]

COLOR_A = "#2E6DA4"
COLOR_B = "#C0504D"
COLOR_RANDOM = "#8C8C8C"
COLOR_ACCENT = "#E8A33D"
COLOR_GRID = "#D9D9D9"


# --------------------------------------------------------------------------
# Data collection and policy evaluation
# --------------------------------------------------------------------------
def collect_data(n_episodes: int = N_EPISODES, seed: int = DATA_SEED) -> list:
    """Collect the course's fixed uniform-random dataset."""
    return collect_random_episodes(n_episodes=n_episodes, seed=seed)


def evaluate_policy(policy, n_episodes: int = EVAL_EPISODES,
                    seed_base: int = EVAL_SEED_BASE) -> dict:
    """Run ``policy`` in the simulator and return the three course metrics.

    Parameters
    ----------
    policy : callable
        ``policy(state_name) -> action_name``.
    n_episodes : int
        Number of fresh evaluation trajectories.
    seed_base : int
        Episode ``i`` uses ``GlucoseEnv(seed=seed_base + i)``.

    Returns
    -------
    dict with
        mean_return : mean undiscounted sum of rewards (no gamma),
        std_return  : population standard deviation of episode returns,
        b3_fraction : total steps spent in B3 / total steps,
        b1_rate     : fraction of episodes terminated by entering B1,
        mean_length : mean number of steps per episode,
        returns     : list of per-episode returns,
        occupancy   : dict state -> fraction of visited states,
        lengths     : list of episode lengths.
    """
    returns = []
    lengths = []
    b3_steps = 0
    total_steps = 0
    b1_terminations = 0
    reset_b1 = 0
    occupancy = Counter()

    for i in range(n_episodes):
        env = GlucoseEnv(seed=seed_base + i)
        s = env.reset()
        if s == "B1":
            reset_b1 += 1
        done = False
        episode_return = 0.0
        steps = 0
        terminated_by_b1 = False

        while not done:
            a = policy(s)
            ns, r, done, _ = env.step(a)
            episode_return += r
            steps += 1
            if ns == "B3":
                b3_steps += 1
            if ns == "B1":
                terminated_by_b1 = True
            occupancy[ns] += 1
            s = ns

        if steps > MAX_STEPS:
            raise RuntimeError(
                f"episode exceeded the course horizon: {steps} > {MAX_STEPS}")
        total_steps += steps
        returns.append(episode_return)
        lengths.append(steps)
        if terminated_by_b1:
            b1_terminations += 1

    occupancy_frac = {
        st: (occupancy.get(st, 0) / total_steps if total_steps else 0.0)
        for st in STATE_NAMES
    }
    return {
        "mean_return": float(statistics.mean(returns)),
        "std_return": float(statistics.pstdev(returns)),
        "b3_fraction": float(b3_steps / total_steps) if total_steps else 0.0,
        "b1_rate": float(b1_terminations / n_episodes),
        "reset_b1_count": int(reset_b1),
        "mean_length": float(statistics.mean(lengths)),
        "returns": [float(x) for x in returns],
        "lengths": [int(x) for x in lengths],
        "occupancy": occupancy_frac,
        "total_steps": int(total_steps),
    }


def make_random_policy(seed: int = RANDOM_ACTION_SEED):
    """Uniform random policy with its own RNG (for a baseline)."""
    rng = random.Random(seed)

    def policy(_state):
        return rng.choice(ACTION_NAMES)

    return policy


# --------------------------------------------------------------------------
# Part A: model estimation + value iteration
# --------------------------------------------------------------------------
def train_model_based(episodes: list, gamma: float = GAMMA) -> ModelBasedAgent:
    agent = ModelBasedAgent(list(STATE_NAMES), list(ACTION_NAMES), gamma=gamma)
    agent.learn_model(episodes)
    agent.run_value_iteration(theta=1e-4, max_iterations=1000)
    return agent


def transition_error(agent: ModelBasedAgent, reference: ModelBasedAgent) -> dict:
    """Compare an estimated model with the full-data model.

    Returns mean / max total-variation distance between P_hat distributions
    and the max absolute difference between reward estimates, over all
    (s, a) pairs.
    """
    tvs = []
    tvs_nonterminal = []
    reward_diffs = []
    for s in STATE_NAMES:
        for a in ACTION_NAMES:
            p = agent.get_transition(s, a)
            q = reference.get_transition(s, a)
            keys = set(p) | set(q)
            tv = 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
            tvs.append(tv)
            if s != "B1":
                tvs_nonterminal.append(tv)
            reward_diffs.append(abs(agent.get_reward(s, a)
                                     - reference.get_reward(s, a)))
    return {
        "mean_tv": float(np.mean(tvs)),
        "max_tv": float(np.max(tvs)),
        "mean_tv_nonterminal": float(np.mean(tvs_nonterminal)),
        "max_tv_nonterminal": float(np.max(tvs_nonterminal)),
        "max_reward_diff": float(np.max(reward_diffs)),
        "mean_reward_diff": float(np.mean(reward_diffs)),
    }


def part_a_experiments(data: list, outdir: Path,
                       eval_episodes: int = EVAL_EPISODES,
                       data_seed: int = DATA_SEED) -> dict:
    """Train Part A on the full budget and on the ablation prefixes."""
    print("\n=== Part A: model estimation + value iteration ===")

    full_agent = train_model_based(data)
    print(f"full model: {full_agent.get_model_summary()}")

    full_eval = evaluate_policy(full_agent.get_policy, n_episodes=eval_episodes)
    print(f"full-data policy: return={full_eval['mean_return']:.3f} "
          f"B3={full_eval['b3_fraction']:.3f} B1={full_eval['b1_rate']:.3f}")

    # Values / policy / Q table / visits for the full model.
    values = {s: full_agent.get_value(s) for s in STATE_NAMES}
    policy = {s: full_agent.get_policy(s) for s in STATE_NAMES}
    q_table = {s: {a: full_agent.get_q_value(s, a) for a in ACTION_NAMES}
               for s in STATE_NAMES}
    visits = {s: {a: full_agent.get_visit_count(s, a) for a in ACTION_NAMES}
              for s in STATE_NAMES}

    # Ablation: first 500 / 1000 / 2000 episodes.
    ablation = []
    for n in ABLATION_SIZES:
        agent = train_model_based(data[:n])
        ev = evaluate_policy(agent.get_policy, n_episodes=eval_episodes)
        err = transition_error(agent, full_agent)
        changed = [s for s in STATE_NAMES
                   if agent.get_policy(s) != full_agent.get_policy(s)]
        ablation.append({
            "n_episodes": n,
            "n_transitions": int(sum(len(e) for e in data[:n])),
            "visited_pairs": int(agent.get_model_summary()["visited_pairs"]),
            "unvisited_pairs": int(agent.get_model_summary()["unvisited_pairs"]),
            "vi_iterations": int(agent.vi_iterations),
            "values": {s: agent.get_value(s) for s in STATE_NAMES},
            "policy": {s: agent.get_policy(s) for s in STATE_NAMES},
            "max_abs_value_diff_vs_full": float(
                max(abs(agent.get_value(s) - full_agent.get_value(s))
                    for s in STATE_NAMES)),
            "policy_changes_vs_full": changed,
            "eval": {k: ev[k] for k in
                     ("mean_return", "std_return", "b3_fraction", "b1_rate",
                      "mean_length")},
            "model_error": err,
            "transition_B3_w": {k: round(v, 4) for k, v in
                                sorted(agent.get_transition("B3", "w").items())},
        })
        print(f"ablation {n:4d}: return={ev['mean_return']:.3f} "
              f"B3={ev['b3_fraction']:.3f} B1={ev['b1_rate']:.3f} "
              f"max|dV|={ablation[-1]['max_abs_value_diff_vs_full']:.3f} "
              f"policy_changes={changed}")

    random_eval = evaluate_policy(make_random_policy(), n_episodes=eval_episodes)
    print(f"random baseline: return={random_eval['mean_return']:.3f} "
          f"B3={random_eval['b3_fraction']:.3f} B1={random_eval['b1_rate']:.3f}")

    # Policy rationale: B4, B3, B2 details.
    rationale = {}
    for s in ("B2", "B3", "B4", "B5"):
        rationale[s] = {
            "policy": policy[s],
            "q": q_table[s],
            "visits": visits[s],
            "transitions": {
                a: {k: round(v, 4) for k, v in
                    sorted(full_agent.get_transition(s, a).items())}
                for a in ACTION_NAMES
            },
            "rewards": {a: full_agent.get_reward(s, a) for a in ACTION_NAMES},
        }

    bellman_residual = max(
        abs(full_agent.get_value(s)
            - max(full_agent.get_q_value(s, a) for a in ACTION_NAMES))
        for s in STATE_NAMES if s != "B1"
    )

    # Reproducible check of the never-visited fallback: a one-episode
    # dataset necessarily leaves many (s, a) pairs unvisited.
    tiny_agent = ModelBasedAgent(list(STATE_NAMES), list(ACTION_NAMES),
                                 gamma=GAMMA)
    tiny_agent.learn_model(collect_data(n_episodes=1, seed=data_seed))
    unvisited = [(s, a) for s in STATE_NAMES for a in ACTION_NAMES
                 if tiny_agent.get_visit_count(s, a) == 0]
    fallback_ok = bool(unvisited) and all(
        len(tiny_agent.get_transition(s, a)) == len(STATE_NAMES)
        and all(abs(p - 1.0 / len(STATE_NAMES)) < 1e-12
                for p in tiny_agent.get_transition(s, a).values())
        and tiny_agent.get_reward(s, a) == 0.0
        for s, a in unvisited
    )
    if not fallback_ok:
        raise RuntimeError("unvisited (s, a) fallback check failed")

    summary = {
        "settings": {
            "data_seed": data_seed,
            "n_episodes": len(data),
            "n_transitions": int(sum(len(e) for e in data)),
            "gamma": GAMMA,
            "eval_episodes": eval_episodes,
            "eval_seed_base": EVAL_SEED_BASE,
            "theta": 1e-4,
            "max_iterations": 1000,
        },
        "model": full_agent.get_model_summary(),
        "vi": {
            "iterations": int(full_agent.vi_iterations),
            "delta": float(full_agent.vi_delta),
            "converged": bool(full_agent.vi_converged),
            "bellman_residual": float(bellman_residual),
        },
        "fallback_check": {
            "n_episodes": 1,
            "n_unvisited_pairs": len(unvisited),
            "uniform_and_zero_reward": fallback_ok,
        },
        "values": values,
        "policy": policy,
        "q_table": q_table,
        "visits": visits,
        "eval": {k: full_eval[k] for k in
                 ("mean_return", "std_return", "b3_fraction", "b1_rate",
                  "reset_b1_count", "mean_length", "occupancy",
                  "total_steps")},
        "eval_returns": full_eval["returns"],
        "random_eval": {k: random_eval[k] for k in
                        ("mean_return", "std_return", "b3_fraction",
                         "b1_rate", "reset_b1_count", "mean_length")},
        "ablation": ablation,
        "rationale": rationale,
    }

    _plot_visits(visits, outdir / "fig_visits.png")
    _plot_values_policy(values, q_table, policy, ablation,
                        outdir / "fig_policy_values.png")
    _plot_eval_result(full_eval, outdir / "fig_partA_eval.png",
                      COLOR_A, "Part A (model-based)")
    _plot_ablation(ablation, outdir / "fig_ablation.png")

    return summary


# --------------------------------------------------------------------------
# Part B: tabular Q-Learning and online exploration
# --------------------------------------------------------------------------
def train_q_learning(episodes: list, seed: int = 0) -> QLearningAgent:
    """Fit a tabular Q-Learning agent on the given episodes."""
    agent = QLearningAgent(
        list(STATE_NAMES), list(ACTION_NAMES),
        gamma=GAMMA, alpha=ALPHA,
        epsilon=EPSILON, epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN, seed=seed,
    )
    agent.learn(episodes, mode="fixed")
    return agent


def policy_agreement(policy_a: dict, policy_b: dict) -> tuple:
    """Agreement rate and disagreeing states over B2..B9."""
    disagree = [s for s in NONTERMINAL_STATES if policy_a[s] != policy_b[s]]
    n = len(NONTERMINAL_STATES)
    return (n - len(disagree)) / n, disagree


def moving_average(values, window: int) -> np.ndarray:
    """Centred-free trailing moving average with a full window."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return np.array([])
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="valid")


def run_online_mode(mode: str, seed: int) -> dict:
    """Train one online Q-Learning run with fixed or decaying epsilon.

    ``mode`` is either ``"fixed"`` (epsilon stays at 0.3) or ``"decay"``
    (epsilon <- max(0.05, 0.995 * epsilon) after every episode).
    """
    env = GlucoseEnv(seed=ONLINE_TRAIN_SEED_BASE + seed)
    agent = QLearningAgent(
        list(STATE_NAMES), list(ACTION_NAMES),
        gamma=GAMMA, alpha=ALPHA,
        epsilon=EPSILON, epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN, seed=seed,
    )
    returns, lengths, epsilons = [], [], []

    for _episode in range(ONLINE_EPISODES):
        s = env.reset()
        done = False
        episode_return = 0.0
        steps = 0
        while not done:
            a = agent.epsilon_greedy(s)
            ns, r, done, _ = env.step(a)
            agent.update(s, a, r, ns, done)
            episode_return += r
            steps += 1
            s = ns
        returns.append(float(episode_return))
        lengths.append(int(steps))
        epsilons.append(float(agent.epsilon))
        if mode == "decay":
            agent.decay_epsilon()

    ev = evaluate_policy(agent.get_policy, n_episodes=ONLINE_EVAL_EPISODES,
                         seed_base=ONLINE_EVAL_SEED_BASE)
    return {
        "seed": seed,
        "mode": mode,
        "returns": returns,
        "lengths": lengths,
        "epsilons": epsilons,
        "last50_return": float(np.mean(returns[-ONLINE_WINDOW:])),
        "online_mean_return": float(np.mean(returns)),
        "online_mean_length": float(np.mean(lengths)),
        "epsilon_final": float(agent.epsilon),
        "policy": {s: agent.get_policy(s) for s in STATE_NAMES},
        "q_table": agent.get_q_table(),
        "eval": {k: ev[k] for k in
                 ("mean_return", "std_return", "b3_fraction", "b1_rate",
                  "mean_length")},
    }


def part_b_experiments(data: list, outdir: Path,
                       eval_episodes: int = EVAL_EPISODES,
                       part_a: dict = None) -> dict:
    """Offline Q-Learning, policy comparison, and online epsilon study."""
    print("\n=== Part B: tabular Q-Learning ===")

    agent = train_q_learning(data, seed=0)
    q_eval = evaluate_policy(agent.get_policy, n_episodes=eval_episodes)
    q_table = agent.get_q_table()
    policy_b = {s: agent.get_policy(s) for s in STATE_NAMES}
    agreement, disagreements = policy_agreement(part_a["policy"], policy_b)
    print(f"offline Q policy: {[policy_b[s] for s in STATE_NAMES]}")
    print(f"agreement with Part A: {agreement:.3f}, disagree={disagreements}")
    print(f"Q eval: return={q_eval['mean_return']:.3f} "
          f"B3={q_eval['b3_fraction']:.3f} B1={q_eval['b1_rate']:.3f}")

    # Supplementary data-budget ablation for Q-Learning (mirrors Part A).
    q_ablation = []
    for n in ABLATION_SIZES:
        qn = train_q_learning(data[:n], seed=0)
        ev = evaluate_policy(qn.get_policy, n_episodes=eval_episodes)
        pol = {s: qn.get_policy(s) for s in STATE_NAMES}
        agr, dis = policy_agreement(part_a["policy"], pol)
        q_ablation.append({
            "n_episodes": n,
            "n_transitions": int(sum(len(e) for e in data[:n])),
            "policy": pol,
            "agreement": float(agr),
            "disagreements": dis,
            "eval": {k: ev[k] for k in
                     ("mean_return", "std_return", "b3_fraction", "b1_rate",
                      "mean_length")},
        })
        print(f"Q ablation {n:4d}: agreement={agr:.3f} "
              f"return={ev['mean_return']:.3f} B3={ev['b3_fraction']:.3f}")

    # Online experiments: fixed vs decaying epsilon, five seeds each.
    online_runs = {
        "fixed": [run_online_mode("fixed", s) for s in ONLINE_SEEDS],
        "decay": [run_online_mode("decay", s) for s in ONLINE_SEEDS],
    }
    reference_return = max(part_a["eval"]["mean_return"],
                           q_eval["mean_return"])
    online_summary = {}
    online_curves = {}
    online_regret = {}
    for mode, runs in online_runs.items():
        returns_matrix = np.array([r["returns"] for r in runs], dtype=float)
        eps_matrix = np.array([r["epsilons"] for r in runs], dtype=float)
        ma = np.array([moving_average(r["returns"], ONLINE_WINDOW)
                       for r in runs])
        online_curves[mode] = {
            "mean": ma.mean(axis=0),
            "std": ma.std(axis=0),
        }
        regret = np.array([
            np.cumsum(reference_return - np.asarray(r["returns"], dtype=float))
            for r in runs
        ])
        online_regret[mode] = {
            "mean": regret.mean(axis=0),
            "std": regret.std(axis=0),
        }
        ev_mean = {
            k: float(np.mean([r["eval"][k] for r in runs]))
            for k in ("mean_return", "b3_fraction", "b1_rate", "mean_length")
        }
        ev_std = {
            k: float(np.std([r["eval"][k] for r in runs]))
            for k in ("mean_return", "b3_fraction", "b1_rate")
        }
        online_summary[mode] = {
            "windows": len(ma[0]),
            "last50_return_mean": float(np.mean(
                [r["last50_return"] for r in runs])),
            "last50_return_std": float(np.std(
                [r["last50_return"] for r in runs])),
            "online_mean_return": float(returns_matrix[:, -ONLINE_WINDOW:].mean()),
            "online_mean_length": float(np.mean(
                [r["online_mean_length"] for r in runs])),
            "epsilon_final_mean": float(np.mean(
                [r["epsilon_final"] for r in runs])),
            "eval_mean": ev_mean,
            "eval_std": ev_std,
            "cumulative_regret_final_mean": float(regret[:, -1].mean()),
            "policies": [r["policy"] for r in runs],
            "per_seed": [
                {
                    "seed": r["seed"],
                    "last50_return": r["last50_return"],
                    "eval_return": r["eval"]["mean_return"],
                    "eval_b3": r["eval"]["b3_fraction"],
                    "eval_b1": r["eval"]["b1_rate"],
                    "epsilon_final": r["epsilon_final"],
                }
                for r in runs
            ],
        }
        print(f"online {mode:5s}: last50={online_summary[mode]['last50_return_mean']:.2f} "
              f"greedy_eval={ev_mean['mean_return']:.2f} "
              f"B3={ev_mean['b3_fraction']:.3f} "
              f"regret={online_summary[mode]['cumulative_regret_final_mean']:.0f}")

    _plot_q_table(q_table, policy_b, outdir / "fig_q_table.png")
    _plot_eval_result(q_eval, outdir / "fig_partB_eval.png",
                      COLOR_B, "Part B (Q-Learning)")
    _plot_comparison(part_a["eval"], q_eval, part_a["random_eval"],
                     outdir / "fig_comparison.png")
    _plot_policy_comparison(part_a["policy"], policy_b, q_ablation,
                            outdir / "fig_policy_comparison.png")
    _plot_online_learning(online_runs, online_curves, reference_return,
                          outdir / "fig_online_learning.png")
    _plot_online_regret(online_runs, online_curves, online_regret,
                        reference_return, outdir / "fig_online_regret.png")

    summary = {
        "settings": {
            "gamma": GAMMA,
            "alpha": ALPHA,
            "epsilon": EPSILON,
            "epsilon_decay": EPSILON_DECAY,
            "epsilon_min": EPSILON_MIN,
            "online_episodes": ONLINE_EPISODES,
            "online_seeds": list(ONLINE_SEEDS),
            "online_window": ONLINE_WINDOW,
            "online_eval_episodes": ONLINE_EVAL_EPISODES,
            "online_eval_seed_base": ONLINE_EVAL_SEED_BASE,
        },
        "q_table": q_table,
        "policy": policy_b,
        "eval": {k: q_eval[k] for k in
                 ("mean_return", "std_return", "b3_fraction", "b1_rate",
                  "mean_length", "occupancy", "total_steps")},
        "agreement": float(agreement),
        "disagreements": disagreements,
        "q_ablation": q_ablation,
        "online": {
            "reference_return": float(reference_return),
            "fixed": online_summary["fixed"],
            "decay": online_summary["decay"],
        },
        "online_curves": {
            mode: {
                "mean": [float(x) for x in online_curves[mode]["mean"]],
                "std": [float(x) for x in online_curves[mode]["std"]],
            }
            for mode in online_curves
        },
    }
    return summary


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _style_axis(ax):
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _plot_visits(visits: dict, path: Path) -> None:
    counts = np.array([[visits[s][a] for a in ACTION_NAMES]
                       for s in STATE_NAMES], dtype=float)
    all_counts = counts.ravel()
    n_sparse = int(np.sum(all_counts < SPARSE_THRESHOLD))
    n_sparse_250 = int(np.sum(all_counts < 250))

    fig = plt.figure(figsize=(12.5, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(counts, cmap="YlOrRd", aspect="auto",
              norm=LogNorm(vmin=max(1.0, counts.min()), vmax=counts.max()))
    ax.set_xticks(range(len(ACTION_NAMES)))
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_yticks(range(len(STATE_NAMES)))
    ax.set_yticklabels(STATE_NAMES)
    ax.set_xlabel("action")
    ax.set_ylabel("state")
    ax.set_title("(a) Visit counts N(s, a)")
    for i in range(len(STATE_NAMES)):
        for j in range(len(ACTION_NAMES)):
            val = int(counts[i, j])
            color = "white" if val > counts.max() * 0.25 else "black"
            ax.text(j, i, f"{val}", ha="center", va="center",
                    fontsize=7.5, color=color)

    # The handout asks for the *distribution* of visit counts (a histogram)
    # in addition to a per-pair view.
    ax = fig.add_subplot(gs[0, 1])
    bins = np.arange(0, 1501, 100)
    ax.hist(all_counts, bins=bins, color=COLOR_A, edgecolor="white")
    ax.axvline(SPARSE_THRESHOLD, color=COLOR_B, linestyle="--", linewidth=1.2)
    ax.set_xlabel("N(s, a)")
    ax.set_ylabel("number of (s, a) pairs")
    ax.set_title("(b) Histogram of visit counts")
    ax.text(0.97, 0.95,
            f"N < 50 : {n_sparse} pairs (all B1)\n"
            f"N < 250: {n_sparse_250} pairs (B1/B8/B9)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec=COLOR_GRID))
    _style_axis(ax)

    ax = fig.add_subplot(gs[1, :])
    pairs = [(s, a, visits[s][a]) for s in STATE_NAMES for a in ACTION_NAMES]
    pairs.sort(key=lambda x: x[2], reverse=True)
    labels = [f"{s}-{a}" for s, a, _ in pairs]
    vals = [n for _, _, n in pairs]
    colors = [COLOR_B if n < SPARSE_THRESHOLD else COLOR_A for n in vals]
    ax.bar(range(len(vals)), vals, color=colors, width=0.78)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("N(s, a), log scale")
    ax.set_title("(c) Sorted counts; red = sparse (N < 50)")
    ax.axhline(SPARSE_THRESHOLD, color=COLOR_B, linestyle="--", linewidth=1)
    _style_axis(ax)
    ax.text(0.98, 0.95, f"sparse pairs: {n_sparse}/36",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=COLOR_GRID))

    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path.name}")


def _plot_values_policy(values: dict, q_table: dict, policy: dict,
                        ablation: list, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    ax = axes[0]
    x = np.arange(len(STATE_NAMES))
    width = 0.26
    sizes = [ab["n_episodes"] for ab in ablation]
    colors = ["#9DC3E6", "#5B9BD5", COLOR_A]
    for k, (n, color) in enumerate(zip(sizes, colors)):
        vals = [ablation[k]["values"][s] for s in STATE_NAMES]
        ax.bar(x + (k - 1) * width, vals, width, label=f"{n} episodes",
               color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES)
    ax.set_ylabel("V(s)")
    ax.set_title("(a) Value estimates for the three data budgets")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)

    ax = axes[1]
    q = np.array([[q_table[s][a] for a in ACTION_NAMES]
                  for s in STATE_NAMES], dtype=float)
    vmax = max(abs(q.min()), abs(q.max()))
    im = ax.imshow(q, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(ACTION_NAMES)))
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_yticks(range(len(STATE_NAMES)))
    ax.set_yticklabels(STATE_NAMES)
    ax.set_xlabel("action")
    ax.set_ylabel("state")
    ax.set_title("(b) Q(s, a); boxed cell = greedy action")
    for i, s in enumerate(STATE_NAMES):
        for j, a in enumerate(ACTION_NAMES):
            chosen = policy[s] == a
            ax.text(j, i, f"{q[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(q[i, j]) > 0.55 * vmax else "black",
                    fontweight="bold" if chosen else "normal")
            if chosen:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="black",
                                           linewidth=2.0))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Q(s, a)")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path.name}")


def _plot_eval_result(eval_result: dict, path: Path, color: str,
                      label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    ax = axes[0]
    returns = np.array(eval_result["returns"])
    ax.hist(returns, bins=24, color=color, alpha=0.85, edgecolor="white")
    ax.axvline(eval_result["mean_return"], color=COLOR_ACCENT,
               linestyle="--", linewidth=1.6,
               label=f"mean = {eval_result['mean_return']:.1f}")
    ax.set_xlabel("undiscounted episode return")
    ax.set_ylabel("episode count")
    ax.set_title(f"(a) {label}: return distribution (500 episodes)")
    ax.legend(frameon=False)
    _style_axis(ax)

    ax = axes[1]
    occ = [eval_result["occupancy"][s] for s in STATE_NAMES]
    colors = [COLOR_ACCENT if s == "B3" else color for s in STATE_NAMES]
    ax.bar(STATE_NAMES, occ, color=colors)
    ax.set_ylabel("fraction of visited states")
    ax.set_title(f"(b) {label}: state occupancy")
    ax.text(0.98, 0.95,
            f"B3 time = {eval_result['b3_fraction']:.1%}\n"
            f"B1 termination = {eval_result['b1_rate']:.1%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=COLOR_GRID))
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path.name}")


def _plot_ablation(ablation: list, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    x = np.arange(len(STATE_NAMES))
    for ab, marker in zip(ablation, ("o", "s", "^")):
        ax.plot(x, [ab["values"][s] for s in STATE_NAMES], marker=marker,
                linewidth=1.4, label=f"{ab['n_episodes']} episodes")
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES)
    ax.set_ylabel("V(s)")
    ax.set_title("(a) Value estimates vs data budget")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)

    ax = axes[1]
    labels = [str(ab["n_episodes"]) for ab in ablation]
    mean_tv = [ab["model_error"]["mean_tv_nonterminal"] for ab in ablation]
    max_tv = [ab["model_error"]["max_tv_nonterminal"] for ab in ablation]
    xx = np.arange(len(labels))
    ax.bar(xx - 0.18, mean_tv, 0.36, color=COLOR_A, label="mean TV (B2-B9)")
    ax.bar(xx + 0.18, max_tv, 0.36, color=COLOR_ACCENT,
           label="max TV (B2-B9)")
    ax.set_xticks(xx)
    ax.set_xticklabels(labels)
    ax.set_xlabel("number of training episodes")
    ax.set_ylabel("total-variation distance to 2000-episode model")
    ax.set_title("(b) Transition-model error (nonterminal pairs)")
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path.name}")


def _plot_q_table(q_table: dict, policy: dict, path: Path) -> None:
    q = np.array([[q_table[s][a] for a in ACTION_NAMES]
                  for s in STATE_NAMES], dtype=float)
    vmax = max(abs(q.min()), abs(q.max()))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    im = ax.imshow(q, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(ACTION_NAMES)))
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_yticks(range(len(STATE_NAMES)))
    ax.set_yticklabels(STATE_NAMES)
    ax.set_xlabel("action")
    ax.set_ylabel("state")
    ax.set_title("Q table after offline Q-Learning (2000 episodes)\n"
                 "boxed cell = greedy action")
    for i, s in enumerate(STATE_NAMES):
        for j, a in enumerate(ACTION_NAMES):
            chosen = policy[s] == a
            ax.text(j, i, f"{q[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(q[i, j]) > 0.55 * vmax else "black",
                    fontweight="bold" if chosen else "normal")
            if chosen:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="black",
                                           linewidth=2.0))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Q(s, a)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path.name}")


def _plot_comparison(a_eval: dict, b_eval: dict, random_eval: dict,
                     path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    labels = ["Part A", "Part B", "Random"]
    colors = [COLOR_A, COLOR_B, COLOR_RANDOM]

    specs = [
        ("mean_return", "mean undiscounted return", "(a) Return", "std_return"),
        ("b3_fraction", "fraction of time in B3", "(b) B3 time", None),
        ("b1_rate", "B1 termination rate", "(c) B1 termination", None),
    ]
    for ax, (key, ylabel, title, err_key) in zip(axes, specs):
        vals = [a_eval[key], b_eval[key], random_eval[key]]
        errs = None
        if err_key:
            errs = [a_eval[err_key], b_eval[err_key], random_eval[err_key]]
        ax.bar(labels, vals, color=colors,
               yerr=errs, capsize=4, error_kw={"linewidth": 1})
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
        _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path.name}")


def _plot_policy_comparison(policy_a: dict, policy_b: dict, q_ablation: list,
                            path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1.0, 1.05]})

    ax = axes[0]
    ax.axis("off")
    cell_text = []
    cell_colors = []
    for s in STATE_NAMES:
        agree = policy_a[s] == policy_b[s]
        cell_text.append([policy_a[s], policy_b[s], "yes" if agree else "no"])
        cell_colors.append([
            "#DCE9F5", "#F6DDD9", "#D9EAD3" if agree else "#F4CCCC"
        ])
    table = ax.table(cellText=cell_text, rowLabels=STATE_NAMES,
                     colLabels=["Part A", "Part B", "agree"],
                     loc="center", cellLoc="center", cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    ax.set_title("(a) Policy comparison (2000 episodes; B1 terminal)")

    ax = axes[1]
    budgets = [ab["n_episodes"] for ab in q_ablation]
    agreement = [ab["agreement"] for ab in q_ablation]
    x = np.arange(len(budgets))
    ax.bar(x, agreement, color=COLOR_B, width=0.55,
           label="agreement with Part A")
    ax.set_ylim(0, 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} episodes" for b in budgets])
    ax.set_ylabel("policy agreement on B2-B9")
    ax.set_title("(b) Q-Learning budget sensitivity")
    for xi, (v, ab) in enumerate(zip(agreement, q_ablation)):
        diff = ",".join(ab["disagreements"]) if ab["disagreements"] else "none"
        ax.text(xi, v + 0.03,
                f"{v:.1%}\ndiff: {diff}\nreturn={ab['eval']['mean_return']:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path.name}")


def _plot_online_learning(online_runs: dict, online_curves: dict,
                          reference_return: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})

    ax = axes[0]
    x = np.arange(ONLINE_WINDOW, ONLINE_EPISODES + 1)
    for mode, color in (("fixed", COLOR_RANDOM), ("decay", COLOR_B)):
        mean = np.asarray(online_curves[mode]["mean"])
        std = np.asarray(online_curves[mode]["std"])
        ax.plot(x, mean, color=color, linewidth=1.6,
                label=f"{mode} epsilon")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
    ax.axhline(reference_return, color=COLOR_A, linestyle="--",
               linewidth=1.0, label="Part A reference")
    ax.set_xlabel("online episode")
    ax.set_ylabel(f"return (moving average, window={ONLINE_WINDOW})")
    ax.set_title("(a) Online learning curve (mean ± std over 5 seeds)")
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)

    ax = axes[1]
    ep = np.arange(ONLINE_EPISODES)
    ax.plot(ep, [EPSILON] * ONLINE_EPISODES, color=COLOR_RANDOM,
            linewidth=1.5, label="fixed")
    decay = [max(EPSILON_MIN, EPSILON * (EPSILON_DECAY ** t)) for t in ep]
    ax.plot(ep, decay, color=COLOR_B, linewidth=1.5, label="decay")
    ax.set_xlabel("online episode")
    ax.set_ylabel("epsilon")
    ax.set_title("(b) Exploration schedules")
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path.name}")


def _plot_online_regret(online_runs: dict, online_curves: dict,
                        online_regret: dict, reference_return: float,
                        path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))

    ax = axes[0]
    x = np.arange(1, ONLINE_EPISODES + 1)
    for mode, color in (("fixed", COLOR_RANDOM), ("decay", COLOR_B)):
        mean = np.asarray(online_regret[mode]["mean"])
        std = np.asarray(online_regret[mode]["std"])
        ax.plot(x, mean, color=color, linewidth=1.6, label=f"{mode} epsilon")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("online episode")
    ax.set_ylabel("cumulative pseudo-regret")
    ax.set_title("(a) Cumulative regret relative to Part A "
                 f"({reference_return:.1f})")
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)

    ax = axes[1]
    modes = ["fixed", "decay"]
    last50 = [np.mean([r["last50_return"] for r in online_runs[m]])
              for m in modes]
    greedy = [np.mean([r["eval"]["mean_return"] for r in online_runs[m]])
              for m in modes]
    last50_std = [np.std([r["last50_return"] for r in online_runs[m]])
                  for m in modes]
    greedy_std = [np.std([r["eval"]["mean_return"] for r in online_runs[m]])
                  for m in modes]
    xx = np.arange(len(modes))
    ax.bar(xx - 0.18, last50, 0.36, yerr=last50_std, capsize=4,
           color=COLOR_RANDOM, label="last 50 online episodes")
    ax.bar(xx + 0.18, greedy, 0.36, yerr=greedy_std, capsize=4,
           color=COLOR_B, label="final greedy policy (500 eval)")
    ax.set_xticks(xx)
    ax.set_xticklabels([f"{m} epsilon" for m in modes])
    ax.set_ylabel("mean undiscounted return")
    ax.set_title("(b) Online return vs final greedy evaluation")
    ax.legend(frameon=False, fontsize=8)
    for xi, (a, b) in enumerate(zip(last50, greedy)):
        ax.text(xi - 0.18, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.18, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path.name}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("a", "b", "all"), default="all",
                        help="which part(s) to run (default: all)")
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                        help="number of training episodes, default 2000")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES,
                        help="number of evaluation episodes, default 500")
    parser.add_argument("--seed", type=int, default=DATA_SEED,
                        help="dataset seed, default 2026")
    parser.add_argument("--outdir", type=Path, default=HERE,
                        help="output directory (default: assignment3/)")
    args = parser.parse_args()

    outdir: Path = args.outdir
    figdir = outdir / "figures"
    resdir = outdir / "results"
    figdir.mkdir(parents=True, exist_ok=True)
    resdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
    })

    print(f"collecting {args.episodes} training episodes (seed={args.seed}) ...")
    data = collect_data(n_episodes=args.episodes, seed=args.seed)
    print(f"dataset: {len(data)} episodes, "
          f"{sum(len(e) for e in data)} transitions")

    results = {}
    if args.part in ("a", "all"):
        results["part_a"] = part_a_experiments(
            data, figdir, eval_episodes=args.eval_episodes,
            data_seed=args.seed)

    if args.part in ("b", "all"):
        if "part_a" not in results:
            # Part B's policy comparison needs the Part A policy; running
            # Part A first also gives the evaluation/reference numbers.
            results["part_a"] = part_a_experiments(
                data, figdir, eval_episodes=args.eval_episodes,
                data_seed=args.seed)
        results["part_b"] = part_b_experiments(
            data, figdir, eval_episodes=args.eval_episodes,
            part_a=results["part_a"])

    # Strip non-serialisable helper keys before saving.
    serialisable = json.loads(json.dumps(results, default=str))
    out_json = resdir / "results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
