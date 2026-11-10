class EventRateController:
    """Integral controller that keeps the event rate near a target."""

    def __init__(self, target_rate=18000.0, kp=1e-4, ki=2e-5):
        self.target_rate = target_rate
        self.kp = kp
        self.ki = ki
        self.integral = 0.0

    def update(self, measured_rate, dt):
        error = self.target_rate - measured_rate
        self.integral += error * dt
        delta = self.kp * error + self.ki * self.integral
        return float(delta)
