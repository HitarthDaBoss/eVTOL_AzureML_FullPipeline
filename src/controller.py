import numpy as np

class PIDController:
    def __init__(self, p, i, d, dt=0.1, out_limit=10.0):
        self.Kp = np.array(p, dtype=float)
        self.Ki = np.array(i, dtype=float)
        self.Kd = np.array(d, dtype=float)
        self.dt = dt
        self.integral = np.zeros(3, dtype=float)
        self.prev_error = np.zeros(3, dtype=float)
        self.out_limit = out_limit

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0

    def step(self, target, current, vel):
        err = np.array(target) - np.array(current)
        self.integral += err * self.dt
        derivative = (err - self.prev_error) / (self.dt + 1e-8)
        self.prev_error = err.copy()
        out = self.Kp * err + self.Ki * self.integral - self.Kd * vel
        return np.clip(out, -self.out_limit, self.out_limit)
