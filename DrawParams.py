import skia
from Unot import Unot
from typing import Tuple
from colorsys import hls_to_rgb, rgb_to_hls
import numpy as np

loc_scale = 6.0
shape_count = 10

class DrawParams:
    def __init__(self, image_path:str, shape_index:int, rotation:float, starness:float, fillColor:int, strokeColor:int, strokeWidth:float, locationX:float, locationY:float, width:float, height:float, aabb=(32,32)):
        self.image_path = image_path
        self.shape_index = shape_index
        self.rotation = rotation
        self.starness = starness
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
        li = self.from_shape_index(self.shape_index)
        rd = self.rotation
        st = self.from_starness(self.starness)
        x, y = self.from_location(self.locationX, self.locationY, self.aabb)
        w, h  = self.from_location(self.width, self.height, self.aabb)
        return DrawParamsNorm(self.image_path, li, rd, st, self.from_color(self.fillColor), self.from_color(self.strokeColor), self.from_stroke_width(self.strokeWidth),x,y,w,h)

    @classmethod
    def color_to_skia(cls, color_number, saturation, lightness):
        col_norm = float(color_number & 0xFFFFFF) / float(0xFFFFFF)
        col = hls_to_rgb(col_norm, saturation, lightness)
        col = tuple(int(i * 255) for i in col)
        col = skia.Color(*col)
        return col
    
    @classmethod
    def from_shape_index(cls, value:int)->float:
        return value / shape_count # + np.random.uniform(0, (1.0 / shape_count))
    
    @classmethod
    def from_starness(cls, value:float)->float:
        return value
    
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
    def __init__(self, image_path, shape_index:float, rotation:float, starness:float,  fillColor:float, strokeColor:float, strokeWidth:float, locationX:float, locationY:float, width:float, hScale:float, aabb=(32,32)):
        self.image_path = image_path
        self.shape_index = shape_index
        self.rotation = rotation
        self.starness = starness
        self.fillColor = fillColor
        self.strokeColor = strokeColor
        self.strokeWidth = strokeWidth
        self.locationX = locationX
        self.locationY = locationY
        self.width = width
        self.hScale = hScale
        self.aabb = aabb

    def to_drawable(self):
        si = self.to_shape_index(self.shape_index)
        rd = self.rotation
        ins = self.to_starness(self.starness)
        fc = self.to_color(self.fillColor)
        sc = self.to_color(self.strokeColor)
        sw = self.to_stroke_width(self.strokeWidth)
        x, y = self.to_location(self.locationX, self.locationY, self.aabb)
        w, h  = self.to_size(self.width, self.hScale, self.aabb)
        return DrawParams(self.image_path, si, rd, ins, fc, sc, sw,x,y,w,h)
    
    @classmethod  
    def data_description(cls):
        return ["filename", "shapeIndex", "rotation", "starness", "fillColor", "strokeColor", "strokeWidth","x","y","w","hScale"]
    
    @classmethod
    def to_shape_index(cls, value:float)->int:
        return int(value * shape_count)
    
    @classmethod
    def to_starness(cls, value:float)->float:
        cutoff = 0.3
        return 0 if abs(value) < cutoff else value * 1.0 / cutoff
    
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
        return [filename, self.shape_index, self.rotation, self.starness, self.fillColor, self.strokeColor, self.strokeWidth,self.locationX,self.locationY,self.width,self.hScale]

    def __str__(self):
        return f"[(]{self.image_path:.2f}, {self.shape_index:.2f}, rd:{self.rotation:.2f}, st:{self.starness:.2f}, \
                    fCol:{self.fillColor:.2f}, sCol:{self.strokeColor:.2f}, sW:{self.strokeWidth:.2f}, \
                    loc:({self.locationX:.2f},{self.locationY:.2f}), sz:({self.width:.2f},hs{self.hScale:.2f})]"

