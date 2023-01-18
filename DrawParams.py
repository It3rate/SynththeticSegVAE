
class DrawParams:
    def __init__(self, filename, label, fillColor, strokeColor, strokeWidth, location, size):
        self.filename = filename
        self.label = label
        self.fillColor = fillColor
        self.strokeColor = strokeColor
        self.strokeWidth = strokeWidth
        self.location = location
        self.size = size

    @classmethod
    def from_data(cls, data):
        return 
    @classmethod  
    def data_description(cls):
        return ["filename", "label", "fillColor", "strokeColor", "strokeWidth","x","y","w","hScale"]
    
    def csv(self, filename:str):
        return [filename, self.label, self.fillColor, self.strokeColor, self.strokeWidth,self.location.x,self.location.y,self.size.w,self.size.hScale]

    def __str__(self):
        return f"({self.filename}, {self.label}, fCol:{self.fillColor}, sCol:{self.strokeColor}, sW:{self.strokeWidth:.2f},   loc:{self.location}, sz:{self.size})"
