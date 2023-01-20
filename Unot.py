
import numpy as np

class Unot:
    def __init__(self, start, end):
        if isinstance(start, complex):
            self.imag = start.imag
            self.real = end.real
            self.complex = start
        else:
            self.imag = start
            self.real = end
            self.complex = complex(end, start)

    def min_value(self):
        return -self.imag if -self.imag <= self.real else self.real
    def max_value(self):
        return -self.imag if -self.imag > self.real else self.real
    
    def mean_std(self):
        segment = [self.imag, self.real]
        std = np.std(segment)
        mean = (self.real - self.imag) / 2.0 + self.imag
        return [mean, std]
    
    def as_range(self):
        return [-self.imag, self.real]
    
    def int_sample(self):
        return np.random.randint(int(self.min_value()), int(self.max_value()))
    
    def uniform_sample(self):
        return np.random.uniform(self.min_value(), self.max_value())
    
    def normal_sample(self):
        m, s = self.mean_std()
        return np.random.normal(m, s)
    
    def __str__(self):
        return f"({self.imag}j,{self.real})"
