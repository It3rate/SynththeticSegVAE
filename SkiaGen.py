import skia
from DrawParams import DrawParams
from Concept import Concept
from Unot import Unot
from PIL import Image
import csv
import random

class SkiaGen:
    px = 32
    def __init__(self):
        self.surface = skia.Surface(self.px, self.px)


    def draw(self, dp : DrawParams):
        with self.surface as canvas:
            canvas.clear(skia.ColorWHITE)
            canvas.save()
            canvas.translate(self.px / 2.0, self.px / 2.0)
            fill = skia.Paint(
                Color=dp.fillColor,
                Style=skia.Paint.kFill_Style)
            self._drawShape(canvas, dp, fill)

            stroke = skia.Paint(
                AntiAlias=True,
                Style=skia.Paint.Style.kStroke_Style,
                Color=dp.strokeColor,
                StrokeWidth=dp.strokeWidth,
                StrokeCap=skia.Paint.Cap.kRound_Cap)
            self._drawShape(canvas, dp, stroke)
            canvas.restore()
        return self.surface.makeImageSnapshot()

    def _drawShape(self, canvas, dp, paint):
        w = dp.size.w
        h = w * dp.size.hScale
        x = dp.location.x - w / 2.0
        y = dp.location.y - h / 2.0
        #print(x)
        rect = skia.Rect(x, y, x + w, y + h)
        if(dp.label == 0):
            canvas.drawRect(rect, paint)
        elif(dp.label == 1):
            canvas.drawOval(rect, paint)
        else:
            r1 = 6
            r2 = 6 + w
            path = skia.Path()
            path.addArc(rect, -90, 180)
            # path.arcTo(x - r2, y - r1, x + r2, y + r1, 180, -180, False)
            # path.arcTo(x - r1, y - r1, x + r1, y + r1, 0, -180, True)
            path.close()
            canvas.drawPath(path, paint)

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
    def gen_data(cls, count:int, start_index:int = 0):
        gen = SkiaGen()
        concepts = cls.get_concepts()
        #dp = DrawParams("test", skia.ColorYELLOW, skia.ColorBLUE, 3, Size(8,12), Point(10, 5)) 
        #dp = DrawParams("test", skia.Color(0, 136, 0), skia.Color(220, 136, 0), 5, Point(10, 5), Size(8,12)) 
        lst = []
        with open('data/params.csv', 'w', newline='\n') as file:
                writer = csv.writer(file)
                writer.writerow(DrawParams.data_description())
                for i in range(count):
                    index = start_index + i
                    dp = random.choice(concepts).sampleParams()
                    image = gen.draw(dp)
                    filename = f"img_{index}.png"
                    save_path = f"./data/{filename}"
                    image.save(save_path)
                    writer.writerow(dp.csv(filename))
                    if len(lst) < 25:
                        img_pil = Image.open(save_path)
                        lst.append(img_pil)
        return lst