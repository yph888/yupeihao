"""
BME1322 Assignment 3 - Part A: model-based RL (model estimation + value iteration)
==================================================================================

This module implements the ``ModelBasedAgent`` protocol required by the
course handout.  It follows the "learn a model, then plan" route:

1. ``learn_model(episodes)`` estimates the transition model

       P_hat(s' | s, a) = N(s, a, s') / N(s, a)

   and the reward model

       R_hat(s, a) = mean of the rewards observed after taking action ``a``
                     in state ``s``.

   For an (s, a) pair that was never visited, P_hat is uniform over all
   states and R_hat is 0.0 (documented in the report).

2. ``run_value_iteration()`` performs synchronous value iteration on the
   estimated model using the Bellman optimality operator

       V_{k+1}(s) = max_a [ R_hat(s, a)
                            + gamma * sum_s' P_hat(s' | s, a) V_k(s') ].

   B1 is the terminal state of the glucose environment, so V(B1) is fixed
   at 0 and is never bootstrapped through.

3. ``get_policy(s)`` extracts the greedy action

       pi(s) = argmax_a Q_hat(s, a),

   with ties broken by the order of ``actions`` supplied by the caller
   (the course order is ['w', 'g', 'i1', 'i2']).

Only the Python standard library is used; no RL/planning package is
imported.  The environment's internal transition/reward tables are never
read: the model is estimated exclusively from the (s, a, r, s', done)
episodes passed to ``learn_model``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence


class ModelBasedAgent:
    """Model-based RL agent for the 9-state blood-glucose environment.

    Parameters
    ----------
    states : sequence of str
        State names, e.g. ``['B1', ..., 'B9']``.
    actions : sequence of str
        Action names in course order, e.g. ``['w', 'g', 'i1', 'i2']``.
    gamma : float
        Discount factor (0.9 in the course environment).
    seed : int or None
        Kept for protocol compatibility; the estimator and value iteration
        are deterministic, so the seed does not change the result.
    """

    #: The environment terminates immediately when B1 is entered.
    TERMINAL_STATE = "B1"

    def __init__(self, states: Sequence[str], actions: Sequence[str],
                 gamma: float = 0.9, seed: Optional[int] = None):
        self.states: List[str] = list(states)
        self.actions: List[str] = list(actions)
        if not self.states:
            raise ValueError("states must not be empty")
        if not self.actions:
            raise ValueError("actions must not be empty")
        self.gamma: float = float(gamma)
        self.seed = seed
        self.rng = random.Random(seed)  # unused, kept for API compatibility

        n_s, n_a = len(self.states), len(self.actions)
        state_set = set(self.states)

        # Counts: N(s, a), N(s, a, s'), sum of rewards for (s, a).
        self.visit_count: Dict[str, Dict[str, int]] = {
            s: {a: 0 for a in self.actions} for s in self.states
        }
        self.transition_count: Dict[str, Dict[str, Dict[str, int]]] = {
            s: {a: {ns: 0 for ns in self.states} for a in self.actions}
            for s in self.states
        }
        self.reward_sum: Dict[str, Dict[str, float]] = {
            s: {a: 0.0 for a in self.actions} for s in self.states
        }

        # Estimated model (filled by learn_model).
        self.transition_model: Dict[str, Dict[str, Dict[str, float]]] = {
            s: {a: {} for a in self.actions} for s in self.states
        }
        self.reward_model: Dict[str, Dict[str, float]] = {
            s: {a: 0.0 for a in self.actions} for s in self.states
        }
        self._model_learned = False

        # Planning results (filled by run_value_iteration).
        self.values: Dict[str, float] = {s: 0.0 for s in self.states}
        self.q_values: Dict[str, Dict[str, float]] = {
            s: {a: 0.0 for a in self.actions} for s in self.states
        }
        self.policy: Dict[str, str] = {s: self.actions[0] for s in self.states}
        self.vi_iterations: int = 0
        self.vi_delta: float = 0.0
        self.vi_converged: bool = False
        self._planning_done = False

        # Expose dimensions for convenience / testing.
        self.n_states = n_s
        self.n_actions = n_a
        self._state_set = state_set

    # ------------------------------------------------------------------
    # Model estimation
    # ------------------------------------------------------------------
    def reset_model(self) -> None:
        """Clear all counts and previously estimated models."""
        for s in self.states:
            for a in self.actions:
                self.visit_count[s][a] = 0
                self.reward_sum[s][a] = 0.0
                for ns in self.states:
                    self.transition_count[s][a][ns] = 0
                self.transition_model[s][a] = {}
                self.reward_model[s][a] = 0.0
        self._model_learned = False
        self._planning_done = False

    def learn_model(self, episodes: list) -> None:
        """Estimate P_hat and R_hat from a list of episodes.

        Each episode is a list of 5-tuples
        ``(state, action, reward, next_state, done)``.

        Unvisited (s, a) pairs receive a uniform next-state distribution
        and R_hat = 0.0.  This keeps the model well-defined and avoids
        dividing by zero; the report discusses the consequence.
        """
        self.reset_model()

        for episode in episodes:
            for transition in episode:
                if len(transition) != 5:
                    raise ValueError(
                        "each transition must be (state, action, reward, "
                        "next_state, done)")
                s, a, r, ns, _done = transition
                if s not in self._state_set:
                    raise ValueError(f"unknown state in data: {s!r}")
                if ns not in self._state_set:
                    raise ValueError(f"unknown next state in data: {ns!r}")
                if a not in self.actions:
                    raise ValueError(f"unknown action in data: {a!r}")

                self.visit_count[s][a] += 1
                self.transition_count[s][a][ns] += 1
                self.reward_sum[s][a] += float(r)

        # Turn counts into probabilities / mean rewards.
        uniform = 1.0 / self.n_states
        for s in self.states:
            for a in self.actions:
                n = self.visit_count[s][a]
                if n == 0:
                    # Never-visited pair: uniform model, zero reward.
                    self.transition_model[s][a] = {
                        ns: uniform for ns in self.states
                    }
                    self.reward_model[s][a] = 0.0
                else:
                    counts = self.transition_count[s][a]
                    self.transition_model[s][a] = {
                        ns: counts[ns] / n for ns in self.states if counts[ns] > 0
                    }
                    self.reward_model[s][a] = self.reward_sum[s][a] / n

        self._model_learned = True

    # ------------------------------------------------------------------
    # Planning: synchronous value iteration
    # ------------------------------------------------------------------
    def _q_from_values(self, s: str, a: str,
                       values: Dict[str, float]) -> float:
        """Q(s, a) = R_hat(s, a) + gamma * sum_s' P_hat(s'|s,a) V(s')."""
        r = self.reward_model[s][a]
        expected_next = 0.0
        for ns, p in self.transition_model[s][a].items():
            expected_next += p * values[ns]
        return r + self.gamma * expected_next

    def run_value_iteration(self, theta: float = 1e-4,
                            max_iterations: int = 1000) -> None:
        """Synchronous value iteration until convergence or iteration cap.

        A synchronous sweep computes every new value from the previous
        value table, then replaces the table.  B1 is terminal: its value
        is fixed at 0 and is not updated.

        After convergence the Q table and the greedy policy are extracted.
        """
        if not self._model_learned:
            raise RuntimeError("call learn_model() before run_value_iteration()")
        if theta <= 0:
            raise ValueError("theta must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        values: Dict[str, float] = {s: 0.0 for s in self.states}
        converged = False
        delta = float("inf")
        iteration = 0

        for iteration in range(1, max_iterations + 1):
            new_values: Dict[str, float] = {}
            delta = 0.0
            for s in self.states:
                if s == self.TERMINAL_STATE:
                    # Terminal state: no future return.
                    new_values[s] = 0.0
                    continue
                best = max(self._q_from_values(s, a, values)
                           for a in self.actions)
                new_values[s] = best
                diff = abs(best - values[s])
                if diff > delta:
                    delta = diff
            values = new_values
            if delta < theta:
                converged = True
                break

        self.values = values
        self.vi_iterations = iteration
        self.vi_delta = delta
        self.vi_converged = converged

        # Extract Q(s, a) and the greedy policy for every state.
        for s in self.states:
            for a in self.actions:
                self.q_values[s][a] = self._q_from_values(s, a, values)
            best_action = self.actions[0]
            best_q = self.q_values[s][best_action]
            for a in self.actions[1:]:
                if self.q_values[s][a] > best_q:  # strict: first action wins ties
                    best_q = self.q_values[s][a]
                    best_action = a
            self.policy[s] = best_action

        self._planning_done = True

    # ------------------------------------------------------------------
    # Accessors required by the protocol
    # ------------------------------------------------------------------
    def get_transition(self, state: str, action: str) -> dict:
        """Return the estimated ``{next_state: probability}`` distribution."""
        if not self._model_learned:
            raise RuntimeError("call learn_model() first")
        if state not in self._state_set:
            raise KeyError(f"unknown state: {state!r}")
        if action not in self.actions:
            raise KeyError(f"unknown action: {action!r}")
        # Return a copy so callers cannot mutate the estimated model.
        return dict(self.transition_model[state][action])

    def get_reward(self, state: str, action: str) -> float:
        """Return the estimated mean reward R_hat(s, a)."""
        if not self._model_learned:
            raise RuntimeError("call learn_model() first")
        if state not in self._state_set:
            raise KeyError(f"unknown state: {state!r}")
        if action not in self.actions:
            raise KeyError(f"unknown action: {action!r}")
        return float(self.reward_model[state][action])

    def get_value(self, state: str) -> float:
        """Return V_hat(state) from the last value-iteration run."""
        if not self._planning_done:
            raise RuntimeError("call run_value_iteration() first")
        if state not in self._state_set:
            raise KeyError(f"unknown state: {state!r}")
        return float(self.values[state])

    def get_policy(self, state: str) -> str:
        """Return the greedy action at ``state`` (ties -> first action)."""
        if not self._planning_done:
            raise RuntimeError("call run_value_iteration() first")
        if state not in self._state_set:
            raise KeyError(f"unknown state: {state!r}")
        return self.policy[state]

    # ------------------------------------------------------------------
    # Convenience helpers (not required by the protocol)
    # ------------------------------------------------------------------
    def get_q_value(self, state: str, action: str) -> float:
        """Return Q_hat(state, action) from the last planning run."""
        if not self._planning_done:
            raise RuntimeError("call run_value_iteration() first")
        return float(self.q_values[state][action])

    def get_visit_count(self, state: str, action: str) -> int:
        """Return N(s, a) in the training data."""
        return int(self.visit_count[state][action])

    def get_model_summary(self) -> dict:
        """Small summary useful for reports / debugging."""
        visited = sum(1 for s in self.states for a in self.actions
                      if self.visit_count[s][a] > 0)
        total = self.n_states * self.n_actions
        return {
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "visited_pairs": visited,
            "total_pairs": total,
            "unvisited_pairs": total - visited,
            "total_transitions": sum(self.visit_count[s][a]
                                     for s in self.states for a in self.actions),
            "vi_iterations": self.vi_iterations,
            "vi_delta": self.vi_delta,
            "vi_converged": self.vi_converged,
        }
