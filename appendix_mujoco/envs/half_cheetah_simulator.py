import numpy as np
import torch
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


class HalfCheetahSimulator(HalfCheetahEnv):
    num_states = 18
    num_actions = 6
    t = 0
    horizon = 1e4   # a large number such that 1000 timesteps will not exceed it
    min_t = 1
    max_t = 9
    ctrl_coef = -0.1

    def __init__(self):
        super(HalfCheetahSimulator, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def __repr__(self):
        return "HalfCheetah_Simulator"

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.sim.data.qpos.flat[:1],
        ])

    def step(self, a, dt=5):
        self.frame_skip = int(dt)
        ob, reward, done, info = super().step(a)
        self.t += dt
        done = done or self.t >= self.horizon
        return ob, reward, done, info

    def reset(self):
        self.t = 0
        return super().reset()

    def calc_reward(self, action=np.zeros(6), state=None, prev_state=None, dt=5):
        assert len(action) == 6
        assert state is not None
        reward_run = (state[-1] - prev_state[-1]) / dt
        reward_ctrl = self.ctrl_coef * np.square(action).sum()
        return reward_run + reward_ctrl

    def is_terminal(self, state=None):
        return self.t >= self.horizon

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
        rewards_run = (states[:, 1:, -1] - states[:, :-1, -1]) / dts
        rewards_ctrl = self.ctrl_coef * torch.sum(actions ** 2, dim=2)
        rewards = rewards_run + rewards_ctrl
        return rewards, torch.ones(K, H, dtype=torch.bool, device=self.device)
