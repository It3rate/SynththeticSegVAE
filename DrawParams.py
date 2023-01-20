import skia
from Unot import Unot
from typing import Tuple
from colorsys import hls_to_rgb, rgb_to_hls

loc_scale = 6.0

class DrawParams:
    def __init__(self, image_path:str, label_index:int, fillColor:int, strokeColor:int, strokeWidth:float, locationX:float, locationY:float, width:float, height:float, aabb=(32,32)):
        self.image_path = image_path
        self.label_index = label_index
        self.fillColor = fillColor
        self.skiaFillColor = self.color_to_skia(fillColor, 0.5, 0.8)
        self.strokeColor = strokeColor
        self.skiaStrokeColor = self.color_to_skia(strokeColor,0.2, 0.8)
        self.strokeWidth = strokeWidth
        self.locationX = locationX
        self.locationY = locationY
        self.width = width
        self.height = height
        self.aabb = aabb
        
    def to_normalized(self):
        x, y = self.from_location(self.locationX, self.locationY, self.aabb)
        w, h  = self.from_location(self.width, self.height, self.aabb)
        return DrawParamsNorm(self.image_path, self.label_index, self.from_color(self.fillColor), self.from_color(self.strokeColor), self.from_stroke_width(self.strokeWidth),x,y,w,h)

    @classmethod
    def color_to_skia(cls, color_number, saturation, lightness):
        col_norm = float(color_number & 0xFFFFFF) / float(0xFFFFFF)
        col = hls_to_rgb(col_norm, saturation, lightness)
        col = tuple(int(i * 255) for i in col)
        col = skia.Color(*col)
        return col
    
    @classmethod
    def from_color(cls, value:int)->float:
        return value / float(0xFFFFFF)
    
    @classmethod
    def from_stroke_width(cls, value:float)->float:
        return value
    
    @classmethod
    def from_location(cls, x:float, y:float, aabb)->Tuple[float, float]:
        return x / (float(aabb[0]) * loc_scale), y / (float(aabb[1]) * loc_scale)
    
    @classmethod
    def from_size(cls, width:float, height:float, aabb)->Tuple[float, float]:
        return width/float(aabb[0]), height/width - 1.0
    
    

class DrawParamsNorm:
    def __init__(self, image_path, label_index, fillColor:float, strokeColor:float, strokeWidth:float, locationX:float, locationY:float, width:float, hScale:float, aabb=(32,32)):
        self.image_path = image_path
        self.label_index = label_index
        self.fillColor = fillColor
        self.strokeColor = strokeColor
        self.strokeWidth = strokeWidth
        self.locationX = locationX
        self.locationY = locationY
        self.width = width
        self.hScale = hScale
        self.aabb = aabb

    def to_drawable(self):
        x, y = self.to_location(self.locationX, self.locationY, self.aabb)
        w, h  = self.to_size(self.width, self.hScale, self.aabb)
        return DrawParams(self.image_path, self.label_index, self.to_color(self.fillColor), self.to_color(self.strokeColor), self.to_stroke_width(self.strokeWidth),x,y,w,h)
    
    @classmethod  
    def data_description(cls):
        return ["filename", "label_index", "fillColor", "strokeColor", "strokeWidth","x","y","w","hScale"]
    
    @classmethod
    def to_color(cls, value:float):
        return int(value * 0xFFFFFF)
        #return int((value + 1.0) / 2.0 * 0xFFFFFF)
    
    @classmethod
    def to_stroke_width(cls, value:float):
        return value * 4.0
    
    @classmethod
    def to_location(cls, x:float, y:float, aabb):
        return (float(x *  (aabb[0] / loc_scale)), float(y * (aabb[1] / loc_scale)))
    
    @classmethod
    def to_size(cls, width:float, hScale:float, aabb):
        w = (width * aabb[0])
        return (w, w * (hScale + 1.0) * (aabb[0] / aabb[1]))

    def csv(self, filename:str):
        return [filename, self.label_index, self.fillColor, self.strokeColor, self.strokeWidth,self.locationX,self.locationY,self.width,self.hScale]

    def __str__(self):
        return f"[(]{self.image_path}, {self.label_index}, fCol:{self.fillColor}, sCol:{self.strokeColor}, sW:{self.strokeWidth:.2f}, \
                    loc:({self.locationX},{self.locationY}), sz:({self.width},hs{self.hScale})]"

