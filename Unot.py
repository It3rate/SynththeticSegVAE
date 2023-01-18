
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
    def __str__(self):
        return f"({self.imag}j,{self.real})"
