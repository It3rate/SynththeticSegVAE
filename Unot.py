
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

    def mean_std(self):
        segment = [self.imag, self.real]
        std = np.std(segment)
        mean = (self.real - self.imag) / 2.0 + self.imag
        return np.array([mean, std])
    
    def __str__(self):
        return f"({self.imag}j,{self.real})"
