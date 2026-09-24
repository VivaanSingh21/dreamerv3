"""Procedurally-generated 100-link DMC swimmer with a 4-parameter sinusoid
torque action space, matched to the VIBES/adversarial_dreamer calibration
setup (repo `adversarial_dreamer`, branch `vibes-swimmer-100link-calibration`).
Vector (401-dim) observations, not pixels. `action_repeat` and the episode
length cutoff are baked into the gym wrapper chain itself (same pattern as
`carla.py`'s `frame_skip`), so `run.steps` / the logged x-axis are already in
agent-decision units.

Ported from `dm_control_local/wrapper.py` in `adversarial_dreamer`: only
`DMCWrapper`, `FrameSkipWrapper`, `sinusoid()`, and
`SwimmerCustomActionWrapperTorque` (NOT the PID-tracked, non-`Torque`
`SwimmerCustomActionWrapper` variant) are ported. `FixedLengthEpisodeWrapper`
is reimplemented from `wrappers.py` (it's trivial).
"""
import functools

import embodied
import gym
import numpy as np

from . import from_gym
from .custom_swimmer import swimmer as swimmer_gen


class DMCWrapper(gym.Env):

  def __init__(self, dmc_env):
    self._env = dmc_env
    obs_spec = self._env.observation_spec()
    action_spec = self._env.action_spec()

    # Flatten dict observation
    obs_dim = sum(np.prod(v.shape) for v in obs_spec.values())
    self.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    self.action_space = gym.spaces.Box(
        low=action_spec.minimum,
        high=action_spec.maximum,
        dtype=np.float32
    )

  def reset(self):
    ts = self._env.reset()
    self._last_time_step = ts  # Save for info dict
    return self._flatten_obs(ts.observation)

  def step(self, action):
    ts = self._env.step(action)
    self._last_time_step = ts  # Save for info dict

    reward = ts.reward or 0.0
    done = ts.last()
    obs = self._flatten_obs(ts.observation)

    info = self._build_info(ts)

    return obs, reward, done, info

  def _flatten_obs(self, obs_dict):
    return np.concatenate([v.ravel() for v in obs_dict.values()])

  def _build_info(self, ts):
    physics = self._env.physics
    info = {}
    try:
      info['to_target_distance'] = physics.nose_to_target_dist()
      info['to_target_vector'] = physics.nose_to_target()
      info['joint_angles'] = physics.joints()
      info['body_velocities'] = physics.body_velocities()
    except Exception as e:
      info['warning'] = f"Could not build some info fields: {str(e)}"
    return info


class FrameSkipWrapper(gym.Wrapper):
  """Repeat the same action for `skip` frames."""

  def __init__(self, env, skip=4):
    super().__init__(env)
    self._skip = skip

  def step(self, action):
    total_reward = 0.0
    done = False
    info = {}
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      total_reward += reward
      if done:
        break
    return obs, total_reward, done, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)


def sinusoid(params, t, n_joints):
  """params = [offset, amp, dtheta_dn, dtheta_dt]; t is scalar time;
  n_joints defines the output dimensionality."""
  offset, amp, dtheta_dn, dtheta_dt = params
  n = np.arange(n_joints)
  return offset + amp * np.sin(dtheta_dn * n + dtheta_dt * t)


class SwimmerCustomActionWrapperTorque(gym.Wrapper):
  """Maps a 4-dim [offset, amp, dtheta/dn, dtheta/dt] action directly into
  per-joint torques via `sinusoid()` (no PID tracking, unlike the sibling
  `SwimmerCustomActionWrapper`)."""

  def __init__(self, env, action_scale=np.array([.3, 1., .5 * np.pi, .3 * np.pi / .05]),
               dt=0.05):
    super().__init__(env)
    self.env = env
    self.dt = dt
    self.t = 0.0

    self.num_joints = self.env.action_space.shape[0]

    a_dim = 4  # 4-dimensional actions for offset, amplitude, dtheta/dn, dtheta/dt
    self.action_scale = action_scale
    assert self.action_scale.shape[0] == a_dim

    self.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(a_dim,), dtype=np.float32)
    self.observation_space = self.env.observation_space
    self.reward_range = self.env.reward_range
    self.metadata = self.env.metadata

  def reset(self):
    self.t = 0.0
    timestep = self.env.reset()
    return timestep

  def get_joint_angles(self):
    return self.env._env.physics.joints()

  def step(self, action):  # action = [offset, amp, dtheta_dn, dtheta_dt]
    action = self.action_scale * action
    torques = sinusoid(action, self.t, self.num_joints)
    torques = np.clip(torques, -1, 1)

    obs, reward, done, info = self.env.step(torques)
    self.t += self.dt

    return obs, reward, done, info


class FixedLengthEpisodeWrapper(gym.Wrapper):

  def __init__(self, env, episode_length):
    super().__init__(env)
    self.episode_length = episode_length
    self.current_step = 0

  def reset(self, **kwargs):
    self.current_step = 0
    return self.env.reset(**kwargs)

  def step(self, action):
    obs, reward, d, info = self.env.step(action)
    self.current_step += 1
    info['episode_done'] = d
    done = self.current_step >= self.episode_length
    return obs, reward, done, info


def make_swimmer_gym_env(n_links=100, episode_length=100, frame_skip=10):
  """Builds the fully-wrapped gym.Env, mirroring
  `adversarial_dreamer.utils.make_swimmer_env`'s construction order."""
  env = swimmer_gen.swimmer(n_links=n_links)
  env = DMCWrapper(env)
  env = SwimmerCustomActionWrapperTorque(env)  # default action_scale/dt, don't override
  env = FrameSkipWrapper(env, skip=frame_skip)
  env = FixedLengthEpisodeWrapper(env, episode_length)
  return env


class Swimmer100(embodied.Env):

  def __init__(self, task, n_links=100, episode_length=100, frame_skip=10):
    env = make_swimmer_gym_env(
        n_links=n_links, episode_length=episode_length, frame_skip=frame_skip)
    self._env = from_gym.FromGym(env, obs_key='vector')

  @functools.cached_property
  def obs_space(self):
    return self._env.obs_space

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    return self._env.step(action)

  def close(self):
    self._env.close()
