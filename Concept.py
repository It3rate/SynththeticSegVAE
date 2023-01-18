import skia
import random
import numpy as np
from colorsys import hls_to_rgb, rgb_to_hls
from Unot import Unot
from DrawParams import DrawParams
from Utils import Point, SizeAspect

class Concept:

    labels = ["apple", "apple2","orange","box", "banana"]

    def __init__(self, label:str, shapeIndex:int, fillColor:Unot, strokeColor:Unot, strokeWidth:Unot, locationX:Unot, locationY:Unot, sizeW:Unot, sizeHScale:Unot):
        self.label = label
        self.shapeIndex = shapeIndex
        self.gFillColor = fillColor
        self.gStrokeColor = strokeColor
        self.gStrokeWidth = strokeWidth
        self.gLocationX = locationX
        self.gLocationY = locationY
        self.gSizeW = sizeW
        self.gSizeHScale = sizeHScale

    def sampleParams(self):
        hFill = hls_to_rgb(self.sample(self.gFillColor),0.5, 0.8)
        hFill = tuple(int(i * 255) for i in hFill)
        hFill = skia.Color(*hFill)
        hStroke = hls_to_rgb(self.sample(self.gStrokeColor),0.2, 0.8)
        hStroke = tuple(int(i * 255) for i in hStroke)
        hStroke = skia.Color(*hStroke)
        strokeWidth = self.sample(self.gStrokeWidth)
        x = self.sample(self.gLocationX)
        y = self.sample(self.gLocationY)
        w = self.sample(self.gSizeW)
        hScale = self.sample(self.gSizeHScale)
        return DrawParams(self.label, self.shapeIndex, hFill, hStroke, strokeWidth, Point(x, y), SizeAspect(w, hScale))

    def sample(self, num:Unot):
        min = -num.imag if -num.imag <= num.real else num.real
        max = -num.imag if -num.imag > num.real else num.real
        return random.uniform(min, max)
    
    def to_numpy(self):
        return np.array([
        [self.labels.index(self.label), self.shapeIndex],
        [
            self.gFillColor.imag / float(0xFFFFFF),
            self.gStrokeColor.imag / float(0xFFFFFF),
            self.gStrokeWidth.imag,
            self.gLocationX.imag / 32.0,
            self.gLocationY.imag / 32.0,
            self.gSizeW.imag,
            self.gSizeHScale.imag,
        ],
        [
            self.gFillColor.real / float(0xFFFFFF),
            self.gStrokeColor.real / float(0xFFFFFF),
            self.gStrokeWidth.real,
            self.gLocationX.real / 32.0,
            self.gLocationY.real / 32.0,
            self.gSizeW.real,
            self.gSizeHScale.real,
        ]])
    
    @classmethod
    def from_numpy(cls, data):
        label_index = data[0][0]
        label = cls.labels[label_index]
        shapeIndex = data[0][1]
        sg = data[1]
        eg = data[2]
        fillColor = Unot(int(sg[0] * 0xFFFFFF), int(eg[0] * 0xFFFFFF))
        strokeColor = Unot(int(sg[1] * 0xFFFFFF), int(eg[1] * 0xFFFFFF))
        strokeWidth = Unot(sg[2], eg[2])
        locX = Unot(sg[3] * 32.0, eg[3] * 32.0)
        locY = Unot(sg[4] * 32.0, eg[4] * 32.0)
        sizeW = Unot(sg[5], eg[5])
        sizeHScale = Unot(sg[6], eg[6])
        return Concept(label, shapeIndex, fillColor, strokeColor, strokeWidth, locX, locY, sizeW, sizeHScale)