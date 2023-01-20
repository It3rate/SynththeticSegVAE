import skia
import numpy as np
from colorsys import hls_to_rgb, rgb_to_hls
from Unot import Unot
from DrawParams import DrawParams, DrawParamsNorm
from Utils import Point, SizeAspect

class Concept:

    labels = ["full", "apple", "apple2","orange","box", "banana"]

    def __init__(self, label:str, shapeIndex:int, fillColor:Unot, strokeColor:Unot, strokeWidth:Unot, locationX:Unot, locationY:Unot, width:Unot, hScale:Unot):
        self.label = label
        self.label_index = self.labels.index(self.label)
        self.shapeIndex = shapeIndex
        self.fillColor = fillColor
        self.strokeColor = strokeColor
        self.strokeWidth = strokeWidth
        self.locationX = locationX
        self.locationY = locationY
        self.width = width
        self.hScale = hScale

    @classmethod
    def init_as_full_range(cls):
        return Concept("full", 0, Unot(0,1), Unot(0,1), Unot(0,1), Unot(1, 1), Unot(1, 1), Unot(-.2, .6), Unot(.3, .3))
    
    def gen_as_ranges(self):
        return [self.fillColor.as_range(), self.strokeColor.as_range(), self.strokeWidth.as_range(), \
                self.locationX.as_range(), self.locationY.as_range(), self.width.as_range(), self.hScale.as_range()]
    def gen_as_mean_std(self):
        return [self.fillColor.mean_std(), self.strokeColor.mean_std(), self.strokeWidth.mean_std(), \
                self.locationX.mean_std(), self.locationY.mean_std(), self.width.mean_std(), self.hScale.mean_std()]
    
    def gen_uniform_samples(self)->DrawParamsNorm:
        rnd_label_index = self.random_label_index()
        return DrawParamsNorm("", rnd_label_index, self.fillColor.uniform_sample(), self.strokeColor.uniform_sample(), self.strokeWidth.uniform_sample(), \
                self.locationX.uniform_sample(), self.locationY.uniform_sample(), self.width.uniform_sample(), self.hScale.uniform_sample())
    
    def gen_normal_samples(self)->DrawParamsNorm:
        rnd_label_index = self.random_label_index()
        return DrawParamsNorm("", rnd_label_index, self.fillColor.normal_sample(), self.strokeColor.normal_sample(), self.strokeWidth.normal_sample(), \
                self.locationX.normal_sample(), self.locationY.normal_sample(), self.width.normal_sample(), self.hScale.normal_sample())

    @classmethod
    def random_label_index(cls):
        return np.random.randint(1, len(cls.labels))
    
    # @classmethod
    # def symnorms_to_color_range(cls, unot_value:Unot):
    #     start = cls.symnorm_to_color(unot_value.imag)
    #     end = cls.symnorm_to_color(unot_value.real)
    #     return (start, end)
    
    # @classmethod
    # def symnorms_to_stroke_width_range(cls, unot_value:Unot):
    #     start = cls.norm_to_stroke_width(unot_value.imag)
    #     end = cls.norm_to_stroke_width(unot_value.real)
    #     return (start, end)
    
    # @classmethod
    # def norm_to_location_range(cls, horz:Unot, vert:Unot, aabb):
    #     start = cls.norm_to_location(horz.imag, horz.real, aabb)
    #     end = cls.norm_to_location(vert.real, vert.real, aabb)
    #     return (start, end)
    
    # @classmethod
    # def symnorms_to_size_range(cls, unot_width:Unot, unot_hScale:float, aabb):
    #     widths = cls.norm_to_size(unot_width.imag, unot_width.real, aabb)
    #     heights = cls.norm_to_size(unot_hScale.imag, unot_hScale.real, aabb)
    #     return (widths[0], widths[1], heights[0], heights[1])
    

    # def sample_params(self):
    #     hFill = hls_to_rgb(self.sample(self.fillColor),0.5, 0.8)
    #     hFill = tuple(int(i * 255) for i in hFill)
    #     hFill = skia.Color(*hFill)
    #     hStroke = hls_to_rgb(self.sample(self.strokeColor),0.2, 0.8)
    #     hStroke = tuple(int(i * 255) for i in hStroke)
    #     hStroke = skia.Color(*hStroke)
    #     strokeWidth = self.sample(self.strokeWidth)
    #     x = self.sample(self.locationX)
    #     y = self.sample(self.locationY)
    #     w = self.sample(self.width)
    #     hScale = self.sample(self.hScale)
    #     return DrawParams(self.label, self.shapeIndex, hFill, hStroke, strokeWidth, x, y, w, hScale)

    # def sample(self, num:Unot):
    #     min = -num.imag if -num.imag <= num.real else num.real
    #     max = -num.imag if -num.imag > num.real else num.real
    #     return np.random.uniform(min, max)
    
    def to_numpy(self):
        return np.array([
        [self.label_index, self.shapeIndex],
        [
            self.fillColor.imag / float(0xFFFFFF),
            self.strokeColor.imag / float(0xFFFFFF),
            self.strokeWidth.imag,
            self.locationX.imag / 32.0,
            self.locationY.imag / 32.0,
            self.width.imag,
            self.hScale.imag,
        ],
        [
            self.fillColor.real / float(0xFFFFFF),
            self.strokeColor.real / float(0xFFFFFF),
            self.strokeWidth.real,
            self.locationX.real / 32.0,
            self.locationY.real / 32.0,
            self.width.real,
            self.hScale.real,
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