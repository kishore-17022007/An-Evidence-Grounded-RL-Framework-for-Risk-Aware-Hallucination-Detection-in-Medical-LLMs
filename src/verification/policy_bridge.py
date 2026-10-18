"""Explicit compatibility boundary between live workflow and simulator policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.rl.dqn import DQNAgent
from src.rl.hrl import HierarchicalAgent, OptionId, available_options
from src.verification.errors import PolicyCompatibilityError


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    policy_name: str
    observation: tuple[float, ...]
    selection: str
    note: str


class SimulatorPolicyBridge:
    """Use DQN/HRL only with genuine simulator observations, never live claim text.

    Existing checkpoints learned in the MedHallu simulator. Live Europe PMC
    documents cannot be represented by their existing 15-float observation.
    Consequently the live workflow records this incompatibility and does not
    call either learned policy or invent a vector encoding.
    """

    def __init__(
        self,
        *,
        dqn: DQNAgent | None = None,
        hrl: HierarchicalAgent | None = None,
    ) -> None:
        self.dqn = dqn
        self.hrl = hrl

    @classmethod
    def from_checkpoints(
        cls,
        *,
        dqn_checkpoint: Path | None = None,
        hrl_checkpoint_dir: Path | None = None,
        seed: int = 17,
        device: str = "auto",
    ) -> SimulatorPolicyBridge:
        """Load project checkpoints only when their simulator contracts match."""
        from src.rl.dqn import DQNConfig
        from src.rl.environment import MedicalVerificationEnv
        from src.rl.hrl import HRLConfig

        env = MedicalVerificationEnv()
        dqn_agent = None
        hrl_agent = None
        if dqn_checkpoint is not None:
            payload = _load_checkpoint_dict(dqn_checkpoint)
            if payload.get("observation_dim") != env.observation_space.shape[0]:
                raise PolicyCompatibilityError("DQN checkpoint observation dimension mismatch")
            if payload.get("action_dim") != env.action_space.n:
                raise PolicyCompatibilityError("DQN checkpoint action dimension mismatch")
            saved_config = payload.get("config", {})
            dqn_agent = DQNAgent.from_env(
                env,
                DQNConfig(
                    hidden_dims=tuple(saved_config.get("hidden_dims", (128, 128))),
                    device=device,
                ),
                seed=seed,
            )
            dqn_agent.load_checkpoint(dqn_checkpoint, load_optimizer=False)
        if hrl_checkpoint_dir is not None:
            hrl_agent = HierarchicalAgent.from_env(
                env, HRLConfig(device=device), seed=seed
            )
            hrl_agent.load_checkpoints(hrl_checkpoint_dir, load_optimizer=False)
        if dqn_agent is None and hrl_agent is None:
            raise PolicyCompatibilityError("Provide a DQN or HRL checkpoint path")
        return cls(dqn=dqn_agent, hrl=hrl_agent)

    def live_workflow_decision(self, *, observation: Any = None) -> PolicyDecision:
        """Refuse live policy calls unless a true compatible observation is supplied."""
        if observation is None:
            raise PolicyCompatibilityError(
                "DQN/HRL checkpoints accept only the 15-feature simulator observation; "
                "live claims and Europe PMC evidence are not encoded by that state."
            )
        state = np.asarray(observation, dtype=np.float32)
        if state.shape != (15,):
            raise PolicyCompatibilityError(
                f"simulator policy requires observation shape (15,), received {state.shape}"
            )
        if self.hrl is not None:
            options = available_options(state)
            option = self.hrl.manager.select_option(
                state, options, explore=False, epsilon=0.0
            )
            return PolicyDecision(
                "HRL simulator manager",
                tuple(float(value) for value in state),
                OptionId(option).name,
                "Simulator-only option decision; not used by live evidence workflow.",
            )
        if self.dqn is not None:
            action = self.dqn.select_action(state, explore=False, epsilon=0.0)
            return PolicyDecision(
                "DQN simulator policy",
                tuple(float(value) for value in state),
                str(action),
                "Simulator-only action; not used by live evidence workflow.",
            )
        raise PolicyCompatibilityError("No simulator policy has been configured.")


def _load_checkpoint_dict(path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"Simulator checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise PolicyCompatibilityError("Simulator checkpoint payload must be a dictionary")
    return payload