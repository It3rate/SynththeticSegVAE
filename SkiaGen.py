from re import X
from tkinter import Canvas
import skia
import csv
import math
import random
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
        
        if(dp.shape_index < 1):
            r1 = 6
            r2 = 6 + w
            path = skia.Path()
            path.addArc(rect, -90, 180)
            # path.arcTo(x - r2, y - r1, x + r2, y + r1, 180, -180, False)
            # path.arcTo(x - r1, y - r1, x + r1, y + r1, 0, -180, True)
            path.close()
            canvas.drawPath(path, paint)
        elif(dp.shape_index < 2):
            canvas.drawOval(rect, paint)
        elif(dp.shape_index < 5):
            path = self.gen_poly_shape(int(dp.shape_index + 1), dp.starness, dp.rotation, w / 2.0, h / 2.0)
            canvas.drawPath(path, paint)
        else:
            canvas.drawRoundRect(rect, 3, 3, paint)

    def gen_data(self, folder_path:str, count:int, concepts=[], start_index:int = 0):
        path = f'{folder_path}/params.csv'
        Utils.EnsureFolder(path)
        gen = SkiaGen()
        concepts = [Concept.init_as_full_range()] if len(concepts) == 0 else concepts
        lst = []
        with open(path, 'w', newline='\n') as file:
                writer = csv.writer(file)
                writer.writerow(DrawParamsNorm.data_description())
                for i in range(count):
                    index = start_index + i
                    cur_concept = random.choice(concepts)
                    dp = cur_concept.gen_uniform_samples()
                    image = gen.draw(dp.to_drawable())
                    filename = f"img_{index}.png"
                    save_path = f"{folder_path}/{filename}"
                    image.save(save_path)
                    writer.writerow(dp.csv(filename))
                    if len(lst) < 25:
                        img_pil = Image.open(save_path) 
                        lst.append(img_pil)
        return lst

    def gen_poly_shape(self, pointCount, starness, rotation, radiusX, radiusY):
        useStarness = True
        useRotation = True

        hasStarness = abs(starness) > 0.001 and useStarness
        scaled_starness = starness / 12.0
        count = pointCount * 2 if hasStarness else pointCount
        pointsPerStep = 4 if hasStarness else 2
        movesPerStep = pointsPerStep // 2
        values = [0] * (count * 2 + 2)
        moves = [0] * (count + 1)
        orientation = rotation if useRotation else 0
        #orientation += (FlatTop or PackHorizontal) and 1/(pointCount * 2) or 0
        #orientation += PackHorizontal * 0.25 or 0
        path = skia.Path()
        x0, y0 = (0,0)
        step = Utils.pix2 / pointCount
        for i in range(pointCount):
            theta = step * i + orientation * Utils.pix2
            x = math.sin(theta) * radiusX
            y = math.cos(theta) * radiusY
            if i == 0:
                x0 = x
                y0 = y
                path.moveTo(x,y)
            else:
                path.lineTo(x,y)

            if hasStarness:
                radius2x = radiusX + radiusX * scaled_starness 
                radius2y = radiusY + radiusY * scaled_starness 
                theta = step * i + step / 2.0 + orientation * Utils.pix2
                mpRadiusX = math.cos(step / 2.0) * radius2x
                mpRadiusY = math.cos(step / 2.0) * radius2y
                x = math.sin(theta) * (mpRadiusX + mpRadiusX * scaled_starness)
                y = math.cos(theta) * (mpRadiusY + mpRadiusY * scaled_starness)
                path.lineTo(x,y)

        path.lineTo(x0,y0)        
        return path


    @classmethod
    def gen_data_concepts(cls, count:int, concepts:list, start_index:int = 0):
        path = 'data/params.csv'
        Utils.EnsureFolder(path)
        gen = SkiaGen()
        concepts = Concept.get_concepts()
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