import numpy as np

class SensorSuite:
    def __init__(self, cfg_sensors: dict):
        self.gps_std = cfg_sensors.get('gps_noise_std', 1.5)
        self.imu_std = cfg_sensors.get('imu_noise_std', 0.02)
        self.alt_std = cfg_sensors.get('altimeter_noise_std', 0.2)
        self.wind_max = cfg_sensors.get('wind_max', 5.0)

    def gps(self, true_pos):
        return true_pos + np.random.normal(0.0, self.gps_std, size=3)

    def altimeter(self, true_alt):
        return true_alt + np.random.normal(0.0, self.alt_std)

    def imu(self, true_accel, true_gyro):
        a = true_accel + np.random.normal(0.0, self.imu_std, size=3)
        g = true_gyro + np.random.normal(0.0, self.imu_std * 0.1, size=3)
        return a, g

    def wind(self):
        return np.random.uniform(-self.wind_max, self.wind_max, size=3)
