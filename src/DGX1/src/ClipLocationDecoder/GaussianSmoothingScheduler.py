class GaussianSmoothingScheduler:

    def __init__(self, start_value, decay_value, min_value = 0.0):
        self.start_value = start_value
        self.decay_value = decay_value
        self.min_value = min_value

        self.standard_deviation = start_value


    def decay(self):
        self.standard_deviation = max(self.min_value,self.standard_deviation * self.decay_value)
        print(f"Decaying std_dev: {self.standard_deviation}")

    def get_std_dev(self):
        return self.standard_deviation