"""
BME1322 Assignment 3 - Part B: tabular Q-Learning
================================================

This module implements the ``QLearningAgent`` protocol required by the
course handout.  The offline main task is batch Q-Learning over a fixed
dataset: ``learn()`` walks the episodes in the given order and applies one
tabular update per transition.  No action is sampled inside ``learn()``;
the epsilon parameters are only used by the optional online exploration
experiment (``epsilon_greedy`` / ``decay_epsilon``).

Update rule (course Chapter 9):

    Q(s, a) <- (1 - alpha) Q(s, a)
               + alpha [ r + gamma max_a' Q(s', a') ]

For ``done=True`` the target is just ``r`` (no future term).  Q is
initialised to 0 for every state-action pair.  Ties in ``get_policy`` are
broken by the order of ``actions`` supplied by the caller, so the course
order ['w', 'g', 'i1', 'i2'] is respected.

Only the Python standard library is used.  The agent never reads the
environment's transition/reward tables; it only uses the
``(s, a, r, s', done)`` transitions it is given.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence


class QLearningAgent:
    """Tabular Q-Learning agent for the 9-state blood-glucose environment.

    Parameters
    ----------
    states : sequence of str
        State names, e.g. ``['B1', ..., 'B9']``.
    actions : sequence of str
        Action names in course order, e.g. ``['w', 'g', 'i1', 'i2']``.
    gamma : float
        Discount factor (0.9 in the course environment).
    alpha : float
        Constant learning rate for the tabular update (0.5 in the handout).
    epsilon : float
        Initial exploration rate (only used by the online experiment).
    epsilon_decay : float
        Multiplicative decay per episode (only used by the online experiment).
    epsilon_min : float
        Lower bound of epsilon (only used by the online experiment).
    seed : int or None
        Seed for the epsilon-greedy random generator.
    """

    def __init__(self, states: Sequence[str], actions: Sequence[str],
                 gamma: float = 0.9, alpha: float = 0.5,
                 epsilon: float = 0.3, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05,
                 seed: Optional[int] = None):
        self.states: List[str] = list(states)
        self.actions: List[str] = list(actions)
        if not self.states:
            raise ValueError("states must not be empty")
        if not self.actions:
            raise ValueError("actions must not be empty")

        self.gamma: float = float(gamma)
        self.alpha: float = float(alpha)
        self.epsilon: float = float(epsilon)
        self.epsilon_decay: float = float(epsilon_decay)
        self.epsilon_min: float = float(epsilon_min)
        self.seed = seed
        self.rng = random.Random(seed)
        self.mode: str = "fixed"

        # Q table initialised to zero.
        self.q: Dict[str, Dict[str, float]] = {
            s: {a: 0.0 for a in self.actions} for s in self.states
        }
        self._state_set = set(self.states)
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        self.update_count: int = 0

    # ------------------------------------------------------------------
    # Offline batch learning (main task)
    # ------------------------------------------------------------------
    def learn(self, episodes: list, mode: str = "fixed") -> None:
        """Apply tabular Q-Learning updates in the given episode order.

        The method deliberately does **not** sample actions or modify
        epsilon: it only replays the provided transitions through
        ``update()``.  ``mode`` is kept for interface compatibility; the
        offline main task does not use an exploration rate.
        """
        self.mode = mode
        for episode in episodes:
            for transition in episode:
                if len(transition) != 5:
                    raise ValueError(
                        "each transition must be (state, action, reward, "
                        "next_state, done)")
                state, action, reward, next_state, done = transition
                self.update(state, action, reward, next_state, done)

    def update(self, state: str, action: str, reward: float,
               next_state: str, done: bool) -> None:
        """One tabular Q-Learning update.

        ``done=True`` suppresses the bootstrapping term, as required by
        the course protocol.  Note that this covers both entering B1 and
        the 200-step truncation present in the dataset; the caller's
        ``done`` flag is authoritative here.
        """
        if state not in self._state_set:
            raise ValueError(f"unknown state: {state!r}")
        if next_state not in self._state_set:
            raise ValueError(f"unknown next state: {next_state!r}")
        if action not in self.actions:
            raise ValueError(f"unknown action: {action!r}")

        old = self.q[state][action]
        if done:
            target = float(reward)
        else:
            best_next = max(self.q[next_state][a] for a in self.actions)
            target = float(reward) + self.gamma * best_next
        self.q[state][action] = (1.0 - self.alpha) * old + self.alpha * target
        self.update_count += 1

    # ------------------------------------------------------------------
    # Accessors required by the protocol
    # ------------------------------------------------------------------
    def get_q_value(self, state: str, action: str) -> float:
        """Return the current Q(state, action)."""
        if state not in self._state_set:
            raise KeyError(f"unknown state: {state!r}")
        if action not in self.actions:
            raise KeyError(f"unknown action: {action!r}")
        return float(self.q[state][action])

    def get_policy(self, state: str) -> str:
        """Return the greedy action at ``state`` (ties -> first action)."""
        if state not in self._state_set:
            raise KeyError(f"unknown state: {state!r}")
        best_action = self.actions[0]
        best_q = self.q[state][best_action]
        for a in self.actions[1:]:
            if self.q[state][a] > best_q:  # strict: first action wins ties
                best_q = self.q[state][a]
                best_action = a
        return best_action

    # ------------------------------------------------------------------
    # Extra helpers for the optional online experiment
    # ------------------------------------------------------------------
    def epsilon_greedy(self, state: str) -> str:
        """Sample an action with the current epsilon (online experiment)."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.get_policy(state)

    def decay_epsilon(self) -> None:
        """Apply one episode of multiplicative epsilon decay."""
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def reset_q(self) -> None:
        """Reset the Q table to zeros (useful for independent runs)."""
        for s in self.states:
            for a in self.actions:
                self.q[s][a] = 0.0
        self.update_count = 0

    def get_q_table(self) -> Dict[str, Dict[str, float]]:
        """Return a deep copy of the Q table."""
        return {s: {a: float(self.q[s][a]) for a in self.actions}
                for s in self.states}

    def get_max_q(self, state: str) -> float:
        """Return max_a Q(state, a)."""
        if state not in self._state_set:
            raise KeyError(f"unknown state: {state!r}")
        return float(max(self.q[state][a] for a in self.actions))

    def get_summary(self) -> dict:
        """Small summary for reports / debugging."""
        return {
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "updates": self.update_count,
        }
