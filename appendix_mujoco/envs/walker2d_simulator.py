import numpy as np
import torch
from gym.envs.mujoco.walker2d import Walker2dEnv


class Walker2dSimulator(Walker2dEnv):

    num_states = 18
    num_actions = 6
    t = 0
    horizon = 1e4
    min_t = 1
    max_t = 7
    ctrl_coef = -1e-3

    def __init__(self):
        super(Walker2dSimulator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __repr__(self):
        return "Walker2d_Simulator"

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10), qpos[:1]]).ravel()

    def step(self, a, dt=4):
        self.frame_skip = int(dt)
        ob, reward, done, info = super().step(a)
        self.t += dt
        done = done or self.t >= self.horizon
        return ob, reward, done, {}

    def reset(self):
        self.t = 0
        return super().reset()

    def calc_reward(self, action=np.zeros(6), state=None, prev_state=None, dt=4):
        assert len(action) == 6
        if state is None:
            state = self._get_obs()
        alive_bonus = 1.0
        reward = (state[-1] - prev_state[-1]) / dt
        reward += alive_bonus
        reward += self.ctrl_coef * np.square(action).sum()
        return reward

    def is_terminal(self, state=None):
        if state is None:
            state = self._get_obs()
        height, ang = state[:2]
        return not (0.8 < height < 2.0 and -1.0 < ang < 1.0) or self.t >= self.horizon

    def get_time_gap(self, action=np.zeros(6), state=None):
        assert len(action) == 6
        if state is None:
            state = self._get_obs()
        amp = (self.max_t-self.min_t)//2
        return np.clip(round(amp*np.cos(20*np.pi*np.linalg.norm(state[8:-1]))+amp+1), self.min_t, self.max_t)

    def get_time_info(self):
        return self.min_t, self.max_t, self.horizon, False  # min_t, max_t, max time length, is continuous

    def calc_reward_in_batch(self, states, actions, dts):
        K, H, _ = actions.size()
        next_states = states[:, 1:, :]
        heights, angs = next_states[:, :, 0], next_states[:, :, 1]
        masks = (heights > .8) & (heights < 2.) & (torch.abs(angs) < 1.)
        masks = masks.cumprod(dim=-1).bool()
        assert masks.size() == (K, H)
        rewards = torch.zeros(K, H, dtype=torch.float, device=self.device)
        rewards_alive = torch.ones(K, H, dtype=torch.float, device=self.device)
        rewards_run = (states[:, 1:, -1] - states[:, :-1, -1]) / dts
        rewards_ctrl = self.ctrl_coef * torch.sum(actions ** 2, dim=-1)
        rewards[masks] += rewards_alive[masks] + rewards_run[masks] + rewards_ctrl[masks]
        return rewards, masks
