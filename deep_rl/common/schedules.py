from ..core import Schedule

class MultistepSchedule(Schedule):
    def __init__(self, initial_value, steps, total_iteractions):
        self.initial_value = initial_value
        self.steps = steps
        self.total_iteractions = total_iteractions
        self.time = None

    def __call__(self):
        val = self.initial_value
        for i, value in self.steps:
            if self.time >= i:
                val = value
        return val


class LinearSchedule(Schedule):
    def __init__(self, start, end, total_iterations):
        self.start = start
        self.end = end
        self.total_iterations = total_iterations
        self.time = None

    def __call__(self):
        return self.end + (0 if self.time >= self.total_iterations else float(self.start - self.end) * float(self.total_iterations - self.time) / float(self.total_iterations))