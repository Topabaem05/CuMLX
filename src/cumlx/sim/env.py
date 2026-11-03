"""A vectorized environment for one task on one backend, in the shape that rsl_rl's OnPolicyRunner expects."""
from __future__ import annotations

from typing import Any

from ..errors import missing_dependency
from .task import StepContext, Task, require_torch


class TaskEnv:
    def __init__(self, task: Task, backend: Any, *, seed: int = 0, fixed_command: tuple[float, ...] | None = None):
        torch = require_torch()
        self.task, self.backend = task, backend
        self.num_envs: int = backend.num_envs
        self.num_actions = task.num_actions
        self.max_episode_length = task.max_episode_steps
        self.device = backend.device
        self.cfg = {'task': task.name}
        self.fixed_command = fixed_command
        self._generator = torch.Generator().manual_seed(seed)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.commands = torch.zeros((self.num_envs, len(task.command_ranges)), device=self.device)
        self.extras: dict[str, Any] = {}
        self.state: Any = None
        self.obs: Any = None
        self.reset()

    def _context(self, terminated: Any = None) -> StepContext:
        return StepContext(self.state, self.actions, self.last_actions, self.commands, self.task, terminated)

    def _new_commands(self, envs: Any) -> None:
        torch = require_torch()
        if not self.task.command_ranges or len(envs) == 0:
            return
        if self.fixed_command is not None:
            self.commands[envs] = self.commands.new_tensor(self.fixed_command)
            return
        values = torch.rand((len(envs), len(self.task.command_ranges)), generator=self._generator).to(self.device)
        low = self.commands.new_tensor([r[0] for r in self.task.command_ranges])
        high = self.commands.new_tensor([r[1] for r in self.task.command_ranges])
        self.commands[envs] = low + values * (high - low)

    def _reset_envs(self, envs: Any) -> None:
        if len(envs) == 0:
            return
        self.backend.reset(envs, self.task.reset(len(envs), self._generator, self.task))
        self.episode_length_buf[envs] = 0
        self.actions[envs] = 0.0
        self.last_actions[envs] = 0.0
        self._new_commands(envs)

    def reset(self) -> Any:
        torch = require_torch()
        self._reset_envs(torch.arange(self.num_envs, device=self.device))
        self.state = self.backend.state()
        self.obs = self.task.observe(self._context())
        return self.get_observations()

    def get_observations(self) -> Any:
        try:
            from tensordict import TensorDict
        except ModuleNotFoundError:
            raise missing_dependency('tensordict', 'sim', purpose='The rsl_rl observation format') from None
        return TensorDict({'policy': self.obs}, batch_size=[self.num_envs])

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        torch = require_torch()
        task = self.task
        self.actions = torch.clip(actions, -task.action_clip, task.action_clip)
        chosen = self.last_actions if task.action_latency else self.actions
        self.backend.apply(chosen)
        self.backend.step()
        self.state = self.backend.state()
        self.episode_length_buf += 1
        terminated = task.terminated(self._context())
        failed = self.backend.failed_worlds()
        if failed is not None:
            terminated = terminated | failed
        reward = task.reward(self._context(terminated))
        timed_out = self.episode_length_buf >= self.max_episode_length
        done = terminated | timed_out
        self.extras = {'time_outs': timed_out.float(), 'terminated': terminated}
        if task.command_ranges and task.command_seconds:
            every = int(task.command_seconds / task.control_dt)
            self._new_commands(torch.nonzero(self.episode_length_buf % every == 0).flatten())
        self.last_actions = self.actions.clone()
        reset = torch.nonzero(done).flatten()
        self._reset_envs(reset)
        if len(reset):
            self.state = self.backend.state()
        self.obs = task.observe(self._context())
        return self.get_observations(), reward, done, self.extras
