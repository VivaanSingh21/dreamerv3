import functools
import os

import elements
import embodied
import numpy as np

from . import from_dm


class DMCDCS(embodied.Env):
  """Distracting Control Suite (DCS), matched to the VIBES benchmark setup.

  Uses the `distracting-control==0.1.1rc3` package (a gym-registration wrapper
  around Google's original `google-research/distracting_control`), the exact
  same distraction source VIBES uses. We call `distracting_control.suite.load`
  directly with `from_pixels=False` and render the frame ourselves (as
  `embodied/envs/dmc.py` does), so DreamerV3's usual DMC plumbing
  (`from_dm.FromDM` + `ActionRepeat` + dm_control's native 1000-step limit)
  applies unchanged.

  The `easy` difficulty tier bakes in: distractor scale 0.1, the first 4 DAVIS
  videos of the train/val partition, and all three distractor types
  (background + camera + color), dynamic. See `distracting_control/suite_utils.py`
  (`DIFFICULTY_SCALE`, `DIFFICULTY_NUM_VIDEOS`).

  Task string: `<domain>_<task>` e.g. `cheetah_run`, `cup_catch`.

  Requires an EGL-capable container: `MUJOCO_GL=egl` and
  `NVIDIA_DRIVER_CAPABILITIES=all,graphics` (the background distractor uploads a
  texture to the GL context every physics step).
  """

  def __init__(
      self, task, action_repeat=2, size=(64, 64), camera=0,
      difficulty='easy', dynamic=True,
      distraction_types=('background', 'camera', 'color'),
      background_videos='train', davis_path=''):
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    from distracting_control import suite as dc_suite
    if davis_path:
      davis_path = os.path.expanduser(davis_path)
      dc_suite.BG_DATA_PATH = os.path.join(davis_path, 'DAVIS/JPEGImages/480p')

    domain, dmc_task = task.split('_', 1)
    if domain == 'cup':  # Only DMC domain with an underscore in its real name.
      domain = 'ball_in_cup'

    self._dmenv = dc_suite.load(
        domain, dmc_task,
        difficulty=difficulty,
        intensity=None,
        dynamic=dynamic,
        distraction_types=tuple(distraction_types),
        background_dataset_videos=background_videos,
        render_kwargs=dict(camera_id=camera),
        from_pixels=False,
    )
    self._env = from_dm.FromDM(self._dmenv)
    self._env = embodied.wrappers.ActionRepeat(self._env, action_repeat)
    self._size = tuple(size)
    self._camera = camera

  @functools.cached_property
  def obs_space(self):
    basic = ('is_first', 'is_last', 'is_terminal', 'reward')
    spaces = {k: self._env.obs_space[k] for k in basic}
    spaces['image'] = elements.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    basic = ('is_first', 'is_last', 'is_terminal', 'reward')
    obs = {k: obs[k] for k in basic}
    obs['image'] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
    return obs
