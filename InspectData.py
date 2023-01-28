
from skimage import io, transform
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from DrawParams import DrawParams
from SkiaGen import SkiaGen
import numpy as np
from PIL import Image
from IPython.display import display

class InspectData:
    def __init__(self):
        self.skiaGen = SkiaGen()
    
    def getImage(self, dp:DrawParams):
        return self.skiaGen.draw(dp)
    def getImageAtIndex(self, index, dataset):
        dp = dataset[index]
        return self.skiaGen.draw(dp)
    
    def show_images(self, images:list, save_path = "",show=True):
        w = images[0][0].width
        h = images[0][0].height
        n_cols = len(images[0])
        n_rows = len(images)
        width = n_cols*w
        height = n_rows*h
        result = Image.new('RGB', (width, height), (255, 255, 255))

        for i in range(n_rows):
            for j in range(n_cols):
                result.paste(images[i][j], (j*w, i*h))

        if not (save_path == ""):
            result.save(save_path)
            if show:
                plt.imshow(plt.imread(save_path))
                plt.show()

    def show_images_mpl(self, images:list, save_path = ""):
        if not isinstance(images[0], list):
            images = list(images)
        y_count = len(images)
        x_count = len(images[0])
        fig = plt.figure(figsize=(x_count, y_count))
        grid = ImageGrid(fig, 111, nrows_ncols=(y_count, x_count), axes_pad=0.1)
        for j, row in enumerate(images):
            for i, image in enumerate(row):
                index = j * x_count + i
                grid[index].axis('off')
                grid[index].imshow(image)

        if not (save_path == ""):
             plt.savefig(save_path, dpi=300, bbox_inches='tight',pad_inches=0)

        plt.show()


    def show_data(self, batch_sample):
        count = len(batch_sample['label'])
        image_paths = batch_sample['image_path']
        images = list()
        for i, image_path in enumerate(image_paths):
            images.append(io.imread(image_path))
        self.show_images(images)

    def show_batch(self, dataloader):
        for i, sample in enumerate(dataloader):
            #print(i, sample['image_path'], sample['label'], sample['gen'].shape)
            if i == 3:
                self.show_data(sample)
                break
