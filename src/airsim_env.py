import airsim
import numpy as np
import time

class AirSimDrone:
    def __init__(self, ip='127.0.0.1', cfg=None):
        self.cfg = cfg or {}
        port = int(self.cfg.get('airsim', {}).get('port', 41451))
        self.client = airsim.MultirotorClient(ip=ip, port=port)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # flight params
        self.dt = float(self.cfg.get('env', {}).get('dt', 0.1))
        self.max_wait = 5.0

    def reset(self, start_pos=(0,0,-10)):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # move to start position safely
        self.client.moveToPositionAsync(start_pos[0], start_pos[1], start_pos[2], 5).join()
        time.sleep(0.1)
        return self._get_obs()

    def _get_obs(self):
        s = self.client.getMultirotorState().kinematics_estimated
        pos = np.array([s.position.x_val, s.position.y_val, s.position.z_val], dtype=np.float32)
        vel = np.array([s.linear_velocity.x_val, s.linear_velocity.y_val, s.linear_velocity.z_val], dtype=np.float32)
        return np.concatenate([pos, vel])

    def step(self, accel_cmd):
        # accel_cmd: world frame accelerations or desired velocity (we interpret as velocity setpoint)
        # convert to moveByVelocity for dt duration
        vel_target = np.clip(accel_cmd, -50, 50)  # safety clip
        dur = self.dt
        self.client.moveByVelocityAsync(float(vel_target[0]), float(vel_target[1]), float(vel_target[2]), dur).join()
        obs = self._get_obs()
        # compute reward & done based on config (user can add battery, no-fly zones)
        # default simplistic reward
        goal = np.array([75.0, 75.0, -15.0])  # AirSim Z is negative if NED
        pos = obs[:3]
        dist = np.linalg.norm(goal - pos)
        reward = -dist * 0.01 - 0.001*np.linalg.norm(vel_target)
        done = bool(dist < 2.0)
        info = {"dist": float(dist)}
        return obs, reward, done, info
