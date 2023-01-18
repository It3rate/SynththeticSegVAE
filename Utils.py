

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __str__(self):
        return f"Point({self.x:.2f},{self.y:.2f})"

class SizeAspect:
    def __init__(self,w,hScale):
        self.w = w
        self.hScale = hScale
    def __str__(self):
        return f"Size({self.w:.2f},hScale{self.hScale:.2f})"
        