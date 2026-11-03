"""Isaac Lab style task definitions on engines that run on a Mac: Genesis on Metal, and MuJoCo on the CPU.

A `Task` says what the scene is and how to observe, act, reward and end an episode. A backend runs it. `TaskEnv` is the
vectorized environment that rsl_rl's runner expects. These are not Isaac Lab's tasks or assets, and nothing here runs Isaac Sim.

Importing this package needs only NumPy. Genesis, MuJoCo, torch, tensordict, rsl_rl and MLX are read when they are used.
"""
from .compare import compare_trajectories, worst_per_world
from .task import Actuation, Asset, ResetState, State, StepContext, Task, available_tasks, get_task

__all__ = ['Actuation', 'Asset', 'ResetState', 'State', 'StepContext', 'Task', 'available_tasks', 'compare_trajectories', 'get_task', 'worst_per_world']
