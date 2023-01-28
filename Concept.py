
import numpy as np
from colorsys import hls_to_rgb, rgb_to_hls
from Unot import Unot
from DrawParams import DrawParams, DrawParamsNorm
from Utils import Point, SizeAspect

class Concept:

    labels = ["cresent", "oval", "triangle", "rectangle", "pentagon", "rsquare"]

    def __init__(self, label:str, shapeIndex:Unot, rotation:Unot, starness:Unot, fillColor:Unot, strokeColor:Unot, strokeWidth:Unot, 
                        locationX:Unot, locationY:Unot, width:Unot, hScale:Unot):
        self.label = label
        self.shapeIndex = shapeIndex
        self.rotation = rotation
        self.starness = starness
        self.fillColor = fillColor
        self.strokeColor = strokeColor
        self.strokeWidth = strokeWidth
        self.locationX = locationX
        self.locationY = locationY
        self.width = width
        self.hScale = hScale

    @classmethod
    def get_concepts(cls):
        return [
            #                   idx           rotation        starness       fill            stroke        strokeWidth     locX       locY               width          hScale
            Concept("cresent",  Unot(-.0j,.0), Unot(.0j,.0),   Unot(-.8j,.8), Unot(-.6j,.8),  Unot(-.0j,.1), Unot(0j,1.3),  Unot(.2j,.2), Unot(.2j,.2),  Unot(-.3j,.4),    Unot(.2j,.2)),
            Concept("oval",     Unot(-.1j,.1), Unot(0j,0),     Unot(0j,.0),   Unot(-.5j,.6),  Unot(-.2j,.2), Unot(0j,1.3),  Unot(.2j,.2), Unot(.2j,.2),  Unot(-.2j,.45),   Unot(.4j,.4)),
            Concept("triangle", Unot(-.2j,.2), Unot(.6j,.6),   Unot(.1j,.1),  Unot(-.0j,.3),  Unot(-.2j,.3), Unot(0j,1.3),  Unot(.2j,.2), Unot(.2j,.2),  Unot(-.3j,.45),   Unot(2j,.2)),
            Concept("rectangle",Unot(-.3j,.3), Unot(.0j,.0),   Unot(.2j,.2),  Unot(-.2j,.3),  Unot(-.3j,.4), Unot(0j,1.3),  Unot(.2j,.2), Unot(.2j,.2),  Unot(-.35j,.4),   Unot(.1j,.3)),
            Concept("pentagon", Unot(-.4j,.4), Unot(.0j,.0),   Unot(.4j,.4),  Unot(-.8j,1),   Unot(-.4j,.5), Unot(.2j,1.4), Unot(.2j,.2), Unot(.2j,.2),  Unot(-.4j,.5),    Unot(.05j,.05)),
            Concept("rsquare",  Unot(-.5j,.5), Unot(-.2j,.3),  Unot(0j,0),    Unot(-.1j,.1),  Unot(-.5j,.6), Unot(0j,1.2),  Unot(.2j,.2), Unot(.2j,.2),  Unot(-.4j,.5),    Unot(.3j,.3)),
        ]
    
    @classmethod
    def init_as_full_range(cls):
        si = Unot(0,5)
        ro = Unot(0,1)
        st = Unot(1,1)
        fc = Unot(0,1)
        sc = Unot(0,1)
        sw = Unot(0,1)
        xl = Unot(1,1)
        yl = Unot(1,1)
        w = Unot(-.2, .6)
        hs = Unot(.3, .3)
        return Concept("full", si, ro, st, fc, sc, sw, xl, yl, w, hs )
    
    def gen_as_ranges(self):
        return [self.shapeIndex.as_range(), self.rotation.as_range(), self.starness.as_range(), \
                self.fillColor.as_range(), self.fillColor.as_range(), self.strokeColor.as_range(), self.strokeWidth.as_range(), \
                self.locationX.as_range(), self.locationY.as_range(), self.width.as_range(), self.hScale.as_range()]
    def gen_as_mean_std(self):
        return [self.shapeIndex.mean_std(), self.rotation.mean_std(), self.starness.mean_std(), \
                self.fillColor.mean_std(), self.strokeColor.mean_std(), self.strokeWidth.mean_std(), \
                self.locationX.mean_std(), self.locationY.mean_std(), self.width.mean_std(), self.hScale.mean_std()]
    
    def gen_uniform_samples(self)->DrawParamsNorm:
        rnd_shape_index = self.shapeIndex.uniform_sample() # self.random_shape_index()
        return DrawParamsNorm("", rnd_shape_index, self.rotation.uniform_sample(), self.starness.uniform_sample(),
                self.fillColor.uniform_sample(), self.strokeColor.uniform_sample(), self.strokeWidth.uniform_sample(), \
                self.locationX.uniform_sample(), self.locationY.uniform_sample(), self.width.uniform_sample(), self.hScale.uniform_sample())
    
    def gen_normal_samples(self)->DrawParamsNorm:
        rnd_shape_index = self.shapeIndex.normal_sample()# self.random_shape_index()
        return DrawParamsNorm("", rnd_shape_index, 0,0,0, self.fillColor.normal_sample(), self.strokeColor.normal_sample(), self.strokeWidth.normal_sample(), \
                self.locationX.normal_sample(), self.locationY.normal_sample(), self.width.normal_sample(), self.hScale.normal_sample())

    @classmethod
    def random_shape_index(cls):
        return DrawParams.from_shape_index(np.random.randint(0, len(cls.labels)))
    