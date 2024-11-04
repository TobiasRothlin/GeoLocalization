import math

class MovingAverage:

    def __init__(self, window_size):
        self.window_size = window_size
        self.window = []

    def add(self, value):
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def get(self):
        if len(self.window) == 0:
            return 0
        
        average = sum(self.window) / len(self.window)
        if math.isnan(average):
            print("Warning: Moving Average is NaN")
            print("Window: ", self.window)
            print("Window Size: ", len(self.window))
            raise ValueError("Moving Average is NaN")
        return average
    
    def __str__(self):
        return str(self.get())
    