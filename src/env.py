import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .sensors import SensorSuite
from .airsim_env import AirSimDrone

class EVTOLEnv(gym.Env):
    """
    Wrapper: selects between AirSim or lightweight sim via config.
    Observations normalized outside if desired by trainer.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dt = cfg['env'].get('dt', 0.1)
        self.use_airsim = cfg['env'].get('use_airsim', False)

        # obs: [x,y,z, vx, vy, vz, (optional battery state)]
        # We'll use a 6-dim observation by default
        obs_high = np.array([1e3]*6, dtype=np.float32)
        obs_low = -obs_high
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # actions: accelerations or velocity setpoints in 3 axes
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # model outputs [-1,1]

        if self.use_airsim:
            self.client = AirSimDrone(ip=cfg['airsim'].get('airsim_ip', '127.0.0.1'), cfg=cfg)
        else:
            self.sensors = SensorSuite(cfg['sensors'])

    def reset(self, seed=None, options=None):
        # returns observation and info (gymnasium API)
        if self.use_airsim:
            obs = self.client.reset()
        else:
            # create a simple simulated state: pos (0..5), z 10..20
            pos = np.array([np.random.uniform(0,5), np.random.uniform(0,5), np.random.uniform(10,20)])
            vel = np.zeros(3)
            obs = np.concatenate([pos, vel]).astype(np.float32)
        return obs, {}

    def step(self, action):
        # action expected in env action space (already scaled to env)
        if self.use_airsim:
            obs, reward, done, info = self.client.step(action)
        else:
            # simple dynamics: acceleration = clip(action*max_acc)
            max_acc = 5.0
            accel = np.clip(action, -1, 1) * max_acc
            # integrate with dt
            # maintain state internal is not long-lived here; for quick sim we return simple next state
            pos = np.array([np.random.uniform(0,80), np.random.uniform(0,80), np.random.uniform(10,20)])
            vel = accel * self.dt
            obs = np.concatenate([pos, vel]).astype(np.float32)
            # reward: negative distance to goal approx
            goal = np.array([75.0,75.0,15.0])
            dist = np.linalg.norm(goal - pos)
            reward = -dist * 0.01 - np.linalg.norm(action)*0.01
            done = bool(dist < 2.0)
            info = {"dist": float(dist)}
        return obs, reward, done, info

    def render(self, mode='human'):
        pass
