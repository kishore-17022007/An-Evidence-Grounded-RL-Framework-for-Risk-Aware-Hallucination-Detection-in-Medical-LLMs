"""Options-based hierarchical DQN using an SMDP manager and DQN workers."""

from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.rl.action_space import MedicalAction
from src.rl.dqn import DQNAgent, DQNConfig, QNetwork, Transition, resolve_device


class OptionId(IntEnum):
    """Learned temporally extended skills controlled by the option manager."""

    EVIDENCE_GATHERING = 0
    CONTRADICTION_INVESTIGATION = 1
    VERIFICATION_ASSESSMENT = 2
    FINALIZATION = 3


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """Formal option: initiation predicate, worker action set, and duration bound."""

    option_id: OptionId
    name: str
    primitive_actions: tuple[MedicalAction, ...]
    max_duration: int


OPTION_SPECS: dict[OptionId, OptionSpec] = {
    OptionId.EVIDENCE_GATHERING: OptionSpec(
        OptionId.EVIDENCE_GATHERING,
        "evidence_gathering",
        (
            MedicalAction.RETRIEVE_EVIDENCE,
            MedicalAction.REQUEST_INDEPENDENT_SOURCE,
        ),
        2,
    ),
    OptionId.CONTRADICTION_INVESTIGATION: OptionSpec(
        OptionId.CONTRADICTION_INVESTIGATION,
        "contradiction_investigation",
        (
            MedicalAction.INVESTIGATE_CONTRADICTION,
            MedicalAction.RETRIEVE_EVIDENCE,
        ),
        2,
    ),
    OptionId.VERIFICATION_ASSESSMENT: OptionSpec(
        OptionId.VERIFICATION_ASSESSMENT,
        "verification_assessment",
        (
            MedicalAction.VERIFY_CURRENT_CLAIM,
            MedicalAction.RETRIEVE_EVIDENCE,
        ),
        2,
    ),
    OptionId.FINALIZATION: OptionSpec(
        OptionId.FINALIZATION,
        "finalization",
        (
            MedicalAction.FINALIZE_SUPPORTED,
            MedicalAction.FINALIZE_CONTRADICTED,
            MedicalAction.FINALIZE_INSUFFICIENT_EVIDENCE,
            MedicalAction.ABSTAIN_OR_ESCALATE,
        ),
        1,
    ),
}


def option_initiable(option_id: OptionId, observation: np.ndarray) -> bool:
    """State-dependent initiation sets using only the public 15-feature state."""
    state = np.asarray(observation, dtype=np.float32)
    if state.shape != (15,):
        raise ValueError(f"option initiation expects a 15-feature observation, got {state.shape}")
    budget_remaining = state[6] > 0.0
    if not budget_remaining:
        return False
    if option_id is OptionId.EVIDENCE_GATHERING:
        return bool(state[2] < 0.999)
    if option_id is OptionId.CONTRADICTION_INVESTIGATION:
        return bool(state[4] < 0.999)
    if option_id is OptionId.VERIFICATION_ASSESSMENT:
        return bool(state[9] == 0.0)
    if option_id is OptionId.FINALIZATION:
        return True
    return False


def available_options(observation: np.ndarray) -> tuple[OptionId, ...]:
    """Return all options whose initiation predicates hold in the given state."""
    return tuple(option_id for option_id in OptionId if option_initiable(option_id, observation))


def switchable_options(
    observation: np.ndarray, previous_option: OptionId | None
) -> tuple[OptionId, ...]:
    """Exclude the just-completed option when alternatives can be initiated.

    If it is the only available option, it remains selectable. This constraint
    uses no hidden label, future information, or additional reward.
    """
    options = available_options(observation)
    alternatives = tuple(option for option in options if option is not previous_option)
    return alternatives or options


def option_termination_reason(
    option_id: OptionId,
    before: np.ndarray,
    after: np.ndarray,
    *,
    duration: int,
    terminated: bool,
    truncated: bool,
) -> str | None:
    """Evaluate state-dependent option termination after a primitive action."""
    if terminated:
        return "environment_terminated"
    if truncated:
        return "environment_truncated_budget"
    next_state = np.asarray(after, dtype=np.float32)
    if option_id is OptionId.EVIDENCE_GATHERING and next_state[2] >= 0.5:
        return "evidence_coverage_threshold"
    if option_id is OptionId.CONTRADICTION_INVESTIGATION and next_state[4] >= 0.5:
        return "contradiction_signal_threshold"
    if option_id is OptionId.VERIFICATION_ASSESSMENT and next_state[9] > before[9]:
        return "verification_completed"
    if option_id is OptionId.FINALIZATION:
        return "finalization_action_completed"
    if duration >= OPTION_SPECS[option_id].max_duration:
        return "maximum_option_duration"
    return None


@dataclass(frozen=True, slots=True)
class SMDPTransition:
    """One option-level transition with aggregated return and option duration."""

    observation: np.ndarray
    option: int
    discounted_return: float
    next_observation: np.ndarray
    duration: int
    terminated: bool
    truncated: bool
    next_options: tuple[int, ...]


def smdp_target(
    discounted_return: torch.Tensor,
    next_option_q: torch.Tensor,
    duration: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Bellman target for options: R + gamma**k max Q, unless episode ended."""
    ended = torch.logical_or(terminated.bool(), truncated.bool())
    continuation = torch.pow(
        torch.as_tensor(gamma, device=duration.device, dtype=torch.float32),
        duration.float(),
    )
    return discounted_return + continuation * next_option_q * (~ended).float()


class SMDPReplayBuffer:
    """Bounded replay for temporally extended manager transitions."""

    def __init__(self, capacity: int, *, seed: int) -> None:
        if capacity < 1:
            raise ValueError("SMDP replay capacity must be positive")
        self._items: deque[SMDPTransition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: SMDPTransition) -> None:
        self._items.append(
            SMDPTransition(
                observation=np.asarray(item.observation, dtype=np.float32).copy(),
                option=int(item.option),
                discounted_return=float(item.discounted_return),
                next_observation=np.asarray(item.next_observation, dtype=np.float32).copy(),
                duration=int(item.duration),
                terminated=bool(item.terminated),
                truncated=bool(item.truncated),
                next_options=tuple(item.next_options),
            )
        )

    def sample(self, batch_size: int) -> list[SMDPTransition]:
        if batch_size < 1 or batch_size > len(self._items):
            raise ValueError(
                f"cannot sample {batch_size} transitions from SMDP replay of size {len(self._items)}"
            )
        return self._rng.sample(list(self._items), batch_size)


@dataclass(frozen=True, slots=True)
class HRLConfig:
    """Manager SMDP-DQN settings; each worker gets its own DQN optimizer."""

    manager_learning_rate: float = 1e-3
    manager_batch_size: int = 32
    manager_replay_capacity: int = 10_000
    manager_gamma: float = 0.99
    manager_epsilon_start: float = 1.0
    manager_epsilon_end: float = 0.05
    manager_epsilon_decay: float = 0.995
    target_update_interval: int = 100
    hidden_dims: tuple[int, ...] = (128, 128)
    worker: DQNConfig = DQNConfig()
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.manager_learning_rate <= 0:
            raise ValueError("manager_learning_rate must be positive")
        if self.manager_batch_size < 1 or self.manager_replay_capacity < self.manager_batch_size:
            raise ValueError("manager replay capacity must be at least manager batch size")
        if not 0 <= self.manager_gamma <= 1:
            raise ValueError("manager_gamma must be in [0, 1]")
        if not 0 <= self.manager_epsilon_end <= self.manager_epsilon_start <= 1:
            raise ValueError("manager epsilons must satisfy 0 <= end <= start <= 1")
        if not 0 < self.manager_epsilon_decay <= 1:
            raise ValueError("manager_epsilon_decay must be in (0, 1]")
        if self.target_update_interval < 1:
            raise ValueError("target_update_interval must be positive")


class OptionManager:
    """Learned Q-network manager over currently initiable temporal options."""

    def __init__(
        self,
        observation_dim: int,
        config: HRLConfig,
        *,
        seed: int,
    ) -> None:
        self.config = config
        self.seed = seed
        self.device = resolve_device(config.device)
        self.online = QNetwork(
            observation_dim,
            len(OptionId),
            hidden_dims=config.hidden_dims,
            seed=seed,
        ).to(self.device)
        self.target = QNetwork(
            observation_dim,
            len(OptionId),
            hidden_dims=config.hidden_dims,
            seed=seed + 1,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.manager_learning_rate)
        self.replay = SMDPReplayBuffer(config.manager_replay_capacity, seed=seed)
        self.environment_steps = 0
        self.optimization_steps = 0
        self._rng = random.Random(seed)

    def select_option(
        self,
        observation: np.ndarray,
        valid_options: tuple[OptionId, ...],
        *,
        explore: bool,
        epsilon: float,
    ) -> OptionId:
        """Select an initiable option from observation only."""
        if not valid_options:
            raise ValueError("manager cannot select from an empty option set")
        if not 0 <= epsilon <= 1:
            raise ValueError("manager epsilon must be in [0, 1]")
        if explore and self._rng.random() < epsilon:
            return self._rng.choice(valid_options)
        state = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device)
        with torch.no_grad():
            values = self.online(state)
            invalid = torch.ones(len(OptionId), dtype=torch.bool, device=self.device)
            invalid[[int(option) for option in valid_options]] = False
            best_index = int(values.masked_fill(invalid, -torch.inf).argmax().item())
        return OptionId(best_index)

    def optimize(self) -> float | None:
        """Apply an SMDP Q-learning update with duration exponent gamma**k."""
        if len(self.replay) < self.config.manager_batch_size:
            return None
        batch = self.replay.sample(self.config.manager_batch_size)
        states = torch.as_tensor(
            np.stack([row.observation for row in batch]), dtype=torch.float32, device=self.device
        )
        options = torch.as_tensor(
            [row.option for row in batch], dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        returns = torch.as_tensor(
            [row.discounted_return for row in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.stack([row.next_observation for row in batch]), dtype=torch.float32, device=self.device
        )
        durations = torch.as_tensor(
            [row.duration for row in batch], dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(
            [row.terminated for row in batch], dtype=torch.bool, device=self.device
        )
        truncated = torch.as_tensor(
            [row.truncated for row in batch], dtype=torch.bool, device=self.device
        )

        current_q = self.online(states).gather(1, options).squeeze(1)
        with torch.no_grad():
            values = self.target(next_states)
            next_q_values: list[torch.Tensor] = []
            for row_index, row in enumerate(batch):
                if row.terminated or row.truncated or not row.next_options:
                    next_q_values.append(torch.zeros((), device=self.device))
                else:
                    allowed = torch.as_tensor(row.next_options, device=self.device, dtype=torch.long)
                    next_q_values.append(values[row_index].index_select(0, allowed).max())
            next_q = torch.stack(next_q_values)
            target = smdp_target(
                returns,
                next_q,
                durations,
                terminated,
                truncated,
                gamma=self.config.manager_gamma,
            )
        loss = F.smooth_l1_loss(current_q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.optimization_steps += 1
        if self.optimization_steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu().item())


class HierarchicalAgent:
    """Manager plus independently initialized and trained masked DQN workers."""

    def __init__(
        self,
        observation_dim: int,
        primitive_action_dim: int,
        config: HRLConfig | None = None,
        *,
        seed: int = 17,
    ) -> None:
        self.config = config or HRLConfig()
        self.seed = seed
        self.observation_dim = observation_dim
        self.primitive_action_dim = primitive_action_dim
        self.manager = OptionManager(observation_dim, self.config, seed=seed)
        self.workers: dict[OptionId, DQNAgent] = {
            option_id: DQNAgent(
                observation_dim,
                primitive_action_dim,
                self.config.worker,
                seed=seed + 100 + int(option_id),
                valid_actions=tuple(int(action) for action in spec.primitive_actions),
            )
            for option_id, spec in OPTION_SPECS.items()
        }

    @classmethod
    def from_env(
        cls, env: Any, config: HRLConfig | None = None, *, seed: int = 17
    ) -> HierarchicalAgent:
        shape = getattr(env.observation_space, "shape", None)
        if not shape or len(shape) != 1:
            raise ValueError("HRL requires one-dimensional observation space")
        if not hasattr(env.action_space, "n"):
            raise ValueError("HRL workers require a discrete primitive action space")
        return cls(int(shape[0]), int(env.action_space.n), config, seed=seed)

    def choose_option(
        self,
        observation: np.ndarray,
        *,
        explore: bool,
        epsilon: float,
    ) -> OptionId:
        return self.manager.select_option(
            observation,
            available_options(observation),
            explore=explore,
            epsilon=epsilon,
        )

    def choose_primitive(
        self,
        option_id: OptionId,
        observation: np.ndarray,
        *,
        explore: bool,
        epsilon: float,
    ) -> int:
        worker = self.workers[option_id]
        return worker.select_action(observation, explore=explore, epsilon=epsilon)

    def train_worker_transition(
        self, option_id: OptionId, transition: Transition
    ) -> float | None:
        worker = self.workers[option_id]
        worker.observe(transition)
        return worker.optimize()

    def save_checkpoints(self, directory: Path, *, extra: dict[str, Any] | None = None) -> None:
        """Save manager and each distinct worker separately with action contracts."""
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "observation_dim": self.observation_dim,
                "option_dim": len(OptionId),
                "config": asdict(self.config),
                "seed": self.seed,
                "optimization_steps": self.manager.optimization_steps,
                "online_state_dict": self.manager.online.state_dict(),
                "target_state_dict": self.manager.target.state_dict(),
                "optimizer_state_dict": self.manager.optimizer.state_dict(),
                "extra": extra or {},
            },
            directory / "manager.pt",
        )
        for option_id, worker in self.workers.items():
            worker.save_checkpoint(
                directory / f"worker_{OPTION_SPECS[option_id].name}.pt",
                extra={
                    "option_id": option_id.name,
                    "worker_initialization": "independent_from_seeded_scratch",
                    **(extra or {}),
                },
            )

    def load_checkpoints(self, directory: Path, *, load_optimizer: bool = True) -> dict[str, Any]:
        manager_path = directory / "manager.pt"
        if not manager_path.is_file():
            raise FileNotFoundError(f"HRL manager checkpoint missing: {manager_path}")
        payload = torch.load(manager_path, map_location=self.manager.device, weights_only=False)
        if payload.get("observation_dim") != self.observation_dim:
            raise ValueError("HRL manager checkpoint observation dimension mismatch")
        if payload.get("option_dim") != len(OptionId):
            raise ValueError("HRL manager checkpoint option dimension mismatch")
        self.manager.online.load_state_dict(payload["online_state_dict"])
        self.manager.target.load_state_dict(payload["target_state_dict"])
        if load_optimizer:
            self.manager.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.manager.optimization_steps = int(payload.get("optimization_steps", 0))
        for option_id, worker in self.workers.items():
            worker_path = directory / f"worker_{OPTION_SPECS[option_id].name}.pt"
            if not worker_path.is_file():
                raise FileNotFoundError(f"HRL worker checkpoint missing: {worker_path}")
            worker_payload = worker.load_checkpoint(worker_path, load_optimizer=load_optimizer)
            option_name = worker_payload.get("extra", {}).get("option_id")
            if option_name != option_id.name:
                raise ValueError(f"worker checkpoint option mismatch at {worker_path}")
        return payload