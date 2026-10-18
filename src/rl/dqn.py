"""Deep Q-Network components for the existing Gymnasium environment."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def resolve_device(device: str = "auto") -> torch.device:
    """Select MPS when available, otherwise CPU; allow explicit supported choice."""
    if device == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")
    selected = torch.device(device)
    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this PyTorch runtime")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if selected.type not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported DQN device: {device!r}")
    return selected


class QNetwork(nn.Module):
    """Feed-forward state-action value network with strict input validation."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (128, 128),
        seed: int = 17,
    ) -> None:
        super().__init__()
        if observation_dim < 1 or action_dim < 1:
            raise ValueError("observation_dim and action_dim must be positive")
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths")

        # fork_rng prevents deterministic initialization from perturbing caller RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            dimensions = (observation_dim, *hidden_dims, action_dim)
            layers: list[nn.Module] = []
            layer_count = len(dimensions) - 1
            for layer_index, (input_dim, output_dim) in enumerate(
                zip(dimensions[:-1], dimensions[1:])
            ):
                layers.append(nn.Linear(input_dim, output_dim))
                if layer_index < layer_count - 1:
                    layers.append(nn.ReLU())
            self.network = nn.Sequential(*layers)
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return Q values for a batch or a single observation."""
        if observations.ndim not in (1, 2):
            raise ValueError("observations must have shape [features] or [batch, features]")
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"expected {self.observation_dim} observation features, "
                f"received {observations.shape[-1]}"
            )
        if not torch.isfinite(observations).all():
            raise ValueError("observations must contain only finite values")
        return self.network(observations.float())


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Validated DQN hyperparameters and target-network update policy."""

    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 20_000
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update_mode: str = "hard"
    target_update_interval: int = 100
    target_tau: float = 0.005
    gradient_clip_norm: float = 10.0
    hidden_dims: tuple[int, ...] = (128, 128)
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1 or self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size >= 1")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if self.target_update_mode not in {"hard", "soft"}:
            raise ValueError("target_update_mode must be 'hard' or 'soft'")
        if self.target_update_interval < 1:
            raise ValueError("target_update_interval must be at least 1")
        if not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must be in (0, 1]")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


@dataclass(frozen=True, slots=True)
class Transition:
    """One policy-visible transition; no target or scenario metadata is stored."""

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool


class ReplayBuffer:
    """Bounded FIFO replay memory with seeded uniform sampling."""

    def __init__(self, capacity: int, *, seed: int = 17) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: Transition) -> None:
        self._items.append(
            Transition(
                observation=np.asarray(transition.observation, dtype=np.float32).copy(),
                action=int(transition.action),
                reward=float(transition.reward),
                next_observation=np.asarray(
                    transition.next_observation, dtype=np.float32
                ).copy(),
                terminated=bool(transition.terminated),
                truncated=bool(transition.truncated),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batch_size > len(self._items):
            raise ValueError(
                f"cannot sample {batch_size} transitions from buffer of size {len(self._items)}"
            )
        return self._rng.sample(list(self._items), batch_size)


def bellman_targets(
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float,
    bootstrap_on_truncation: bool = False,
) -> torch.Tensor:
    """Compute DQN targets; budget truncations are terminal by default.

    The project's Gymnasium truncation represents exhaustion of its intrinsic
    action budget, so it defaults to no bootstrap. Generic time-limit users can
    explicitly set ``bootstrap_on_truncation=True``.
    """
    if bootstrap_on_truncation:
        ended = terminated
    else:
        ended = torch.logical_or(terminated.bool(), truncated.bool())
    return rewards + gamma * next_q_values * (~ended.bool()).float()


class DQNAgent:
    """Online/target Q networks, replay optimization, and epsilon-greedy policy."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: DQNConfig | None = None,
        *,
        seed: int = 17,
        valid_actions: tuple[int, ...] | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._rng = random.Random(seed)
        self.device = resolve_device(self.config.device)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.valid_actions = tuple(range(action_dim)) if valid_actions is None else tuple(valid_actions)
        if not self.valid_actions or len(set(self.valid_actions)) != len(self.valid_actions):
            raise ValueError("valid_actions must be a non-empty sequence without duplicates")
        if any(action < 0 or action >= action_dim for action in self.valid_actions):
            raise ValueError("valid_actions contains an index outside the action space")
        self.policy_network = QNetwork(
            observation_dim,
            action_dim,
            hidden_dims=self.config.hidden_dims,
            seed=seed,
        ).to(self.device)
        self.target_network = QNetwork(
            observation_dim,
            action_dim,
            hidden_dims=self.config.hidden_dims,
            seed=seed + 1,
        ).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.config.learning_rate
        )
        self.replay = ReplayBuffer(self.config.replay_capacity, seed=seed)
        self.environment_steps = 0
        self.optimization_steps = 0

    @classmethod
    def from_env(
        cls,
        env: Any,
        config: DQNConfig | None = None,
        *,
        seed: int = 17,
        valid_actions: tuple[int, ...] | None = None,
    ) -> DQNAgent:
        """Derive network dimensions from a Gymnasium environment's spaces."""
        shape = getattr(env.observation_space, "shape", None)
        if not shape or len(shape) != 1:
            raise ValueError("DQN requires a one-dimensional observation space")
        if not hasattr(env.action_space, "n"):
            raise ValueError("DQN requires a discrete action space")
        return cls(
            int(shape[0]),
            int(env.action_space.n),
            config,
            seed=seed,
            valid_actions=valid_actions,
        )

    def epsilon(self, *, decay: float | None = None) -> float:
        """Return current epsilon after exponential environment-step decay."""
        decay_rate = self.config.epsilon_decay if decay is None else decay
        if not 0 < decay_rate <= 1:
            raise ValueError("epsilon decay must be in (0, 1]")
        return max(
            self.config.epsilon_end,
            self.config.epsilon_start * decay_rate**self.environment_steps,
        )

    def select_action(
        self, observation: np.ndarray, *, explore: bool = True, epsilon: float | None = None
    ) -> int:
        """Choose a valid action using only the supplied policy observation."""
        state = np.asarray(observation, dtype=np.float32)
        if state.shape != (self.observation_dim,):
            raise ValueError(
                f"expected observation shape ({self.observation_dim},), received {state.shape}"
            )
        if not np.isfinite(state).all():
            raise ValueError("observation must contain only finite values")
        chance = self.epsilon() if epsilon is None else epsilon
        if not 0 <= chance <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        if explore and self._rng.random() < chance:
            action = self._rng.choice(self.valid_actions)
        else:
            with torch.no_grad():
                tensor = torch.as_tensor(state, device=self.device)
                values = self.policy_network(tensor)
                invalid = torch.ones(self.action_dim, dtype=torch.bool, device=self.device)
                invalid[list(self.valid_actions)] = False
                action = int(values.masked_fill(invalid, -torch.inf).argmax(dim=-1).item())
        if not 0 <= action < self.action_dim:
            raise RuntimeError(f"policy generated invalid action index: {action}")
        return action

    def observe(self, transition: Transition) -> None:
        """Store one policy-visible transition and advance exploration steps."""
        self.replay.add(transition)
        self.environment_steps += 1

    def optimize(self) -> float | None:
        """Sample replay, apply a Huber-loss Bellman update, and update target."""
        if len(self.replay) < self.config.batch_size:
            return None
        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(
            np.stack([item.observation for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.stack([item.next_observation for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        terminated = torch.as_tensor(
            [item.terminated for item in batch], dtype=torch.bool, device=self.device
        )
        truncated = torch.as_tensor(
            [item.truncated for item in batch], dtype=torch.bool, device=self.device
        )

        current_q = self.policy_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_values = self.target_network(next_states)
            invalid_actions = torch.ones(
                self.action_dim, dtype=torch.bool, device=self.device
            )
            invalid_actions[list(self.valid_actions)] = False
            next_q = next_values.masked_fill(invalid_actions, -torch.inf).max(dim=1).values
            targets = bellman_targets(
                rewards,
                next_q,
                terminated,
                truncated,
                gamma=self.config.gamma,
            )
        loss = F.smooth_l1_loss(current_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        self.optimization_steps += 1
        self._update_target()
        return float(loss.detach().cpu().item())

    def _update_target(self) -> None:
        if self.config.target_update_mode == "hard":
            if self.optimization_steps % self.config.target_update_interval == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())
        else:
            tau = self.config.target_tau
            with torch.no_grad():
                for target_parameter, policy_parameter in zip(
                    self.target_network.parameters(), self.policy_network.parameters()
                ):
                    target_parameter.lerp_(policy_parameter, tau)

    def save_checkpoint(self, path: Path, *, extra: dict[str, Any] | None = None) -> None:
        """Save model, optimizer, dimensions, config, RNG and optional metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_state_dict": self.policy_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "valid_actions": self.valid_actions,
            "config": asdict(self.config),
            "seed": self.seed,
            "environment_steps": self.environment_steps,
            "optimization_steps": self.optimization_steps,
            "python_rng_state": self._rng.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: Path, *, load_optimizer: bool = True) -> dict[str, Any]:
        """Load compatible model state; fail clearly on dimension mismatch."""
        if not path.is_file():
            raise FileNotFoundError(f"DQN checkpoint does not exist: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("observation_dim") != self.observation_dim:
            raise ValueError(
                f"checkpoint observation dimension {payload.get('observation_dim')} "
                f"does not match {self.observation_dim}"
            )
        if payload.get("action_dim") != self.action_dim:
            raise ValueError(
                f"checkpoint action dimension {payload.get('action_dim')} "
                f"does not match {self.action_dim}"
            )
        checkpoint_actions = tuple(payload.get("valid_actions", range(self.action_dim)))
        if checkpoint_actions != self.valid_actions:
            raise ValueError(
                f"checkpoint valid action set {checkpoint_actions} does not match "
                f"{self.valid_actions}"
            )
        self.policy_network.load_state_dict(payload["policy_state_dict"])
        self.target_network.load_state_dict(payload["target_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.environment_steps = int(payload.get("environment_steps", 0))
        self.optimization_steps = int(payload.get("optimization_steps", 0))
        if "python_rng_state" in payload:
            self._rng.setstate(payload["python_rng_state"])
        return payload