from tkinter import Canvas
import skia
import csv
import numpy as np
from PIL import Image
from DrawParams import DrawParams, DrawParamsNorm
from Concept import Concept
from Unot import Unot
from Utils import Utils

class SkiaGen:

    def __init__(self, px=32):
        self.px = px
        self.surface = skia.Surface(self.px, self.px)

    def draw(self, dp : DrawParams):
        with self.surface as canvas:
            canvas.clear(skia.ColorWHITE)
            canvas.save()
            canvas.translate(self.px / 2.0, self.px / 2.0)
            fill = skia.Paint(
                Color=dp.skiaFillColor,
                Style=skia.Paint.kFill_Style)
            self._drawShape(canvas, dp, fill)

            stroke = skia.Paint(
                AntiAlias=True,
                Style=skia.Paint.Style.kStroke_Style,
                Color=dp.skiaStrokeColor,
                StrokeWidth=dp.strokeWidth,
                StrokeCap=skia.Paint.Cap.kRound_Cap)
            self._drawShape(canvas, dp, stroke)
            canvas.restore()
        return self.surface.makeImageSnapshot()

    def _drawShape(self, canvas:skia.Canvas, dp:DrawParams, paint:skia.Paint):
        xDir = 1 if dp.width > 0 else -1
        yDir = 1 if dp.height > 0 else -1
        w = abs(dp.width)
        h = abs(dp.height)
        x = dp.locationX - w/2.0
        y = dp.locationY - h/2.0
        rect = skia.Rect(x, y, x + w, y + h)
        if(dp.label_index <= 3):
            canvas.drawRect(rect, paint)
        elif(dp.label_index <= 6):
            r1 = 6
            r2 = 6 + w
            path = skia.Path()
            path.addArc(rect, -90, 180)
            # path.arcTo(x - r2, y - r1, x + r2, y + r1, 180, -180, False)
            # path.arcTo(x - r1, y - r1, x + r1, y + r1, 0, -180, True)
            path.close()
            canvas.drawPath(path, paint)
        else:
            canvas.drawOval(rect, paint)

    def gen_data(self, folder_path:str, count:int, start_index:int = 0):
        path = f'{folder_path}/params.csv'
        Utils.EnsureFolder(path)
        gen = SkiaGen()
        concept = Concept.init_as_full_range()
        lst = []
        with open(path, 'w', newline='\n') as file:
                writer = csv.writer(file)
                writer.writerow(DrawParamsNorm.data_description())
                for i in range(count):
                    index = start_index + i
                    dp = concept.gen_uniform_samples()
                    image = gen.draw(dp.to_drawable())
                    filename = f"img_{index}.png"
                    save_path = f"{folder_path}/{filename}"
                    image.save(save_path)
                    writer.writerow(dp.csv(filename))
                    if len(lst) < 25:
                        img_pil = Image.open(save_path) 
                        lst.append(img_pil)
        return lst

    @classmethod
    def get_concepts(cls):
        return [
            Concept("apple", 1, Unot(.04j,.02), Unot(-.1j,.2),  Unot(-.5j,2),    Unot(6j,6), Unot(6j,6), Unot(-6j,14), Unot(-1.1j,.9)),
            Concept("apple2", 1, Unot(-.15j,.45), Unot(-.2j,.4), Unot(-.5j,2), Unot(6j,6), Unot(6j,6), Unot(-8j,18), Unot(-1j,.7)),
            Concept("orange", 1, Unot(-.07j,.15), Unot(-.8j,1), Unot(-.2j,1),   Unot(6j,6), Unot(6j,6), Unot(-9j,10), Unot(-.9j,.6)),
            Concept("box", 0, Unot(-.6j,.8), Unot(-.5j,.7),     Unot(-2j,4),    Unot(6j,6), Unot(6j,6), Unot(-10j,20), Unot(-.95j,1.05)),
            Concept("banana", 2, Unot(-.11j,.16), Unot(-.01j,.2), Unot(-.8j,2), Unot(6j,6), Unot(6j,6), Unot(-8j,10), Unot(-1.5j,2.5))
        ]
    

    @classmethod
    def gen_dataX(cls, count:int, start_index:int = 0):
        path = 'data/params.csv'
        Utils.EnsureFolder(path)
        gen = SkiaGen()
        concepts = cls.get_concepts()
        #dp = DrawParams("test", skia.ColorYELLOW, skia.ColorBLUE, 3, Size(8,12), Point(10, 5)) 
        #dp = DrawParams("test", skia.Color(0, 136, 0), skia.Color(220, 136, 0), 5, Point(10, 5), Size(8,12)) 
        lst = []
        with open(path, 'w', newline='\n') as file:
                writer = csv.writer(file)
                writer.writerow(DrawParams.data_description())
                for i in range(count):
                    index = start_index + i
                    dp = np.random.choice(concepts).sample_params()
                    image = gen.draw(dp)
                    filename = f"img_{index}.png"
                    save_path = f"./data/{filename}"
                    image.save(save_path)
                    writer.writerow(dp.csv(filename))
                    if len(lst) < 25:
                        img_pil = Image.open(save_path)
                        lst.append(img_pil)
        return lst