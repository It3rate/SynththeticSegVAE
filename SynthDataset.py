
from __future__ import print_function, division
import os
from typing import Type, Any, Optional
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from DrawParams import DrawParams
from SkiaGen import SkiaGen
from Utils import Point, SizeAspect

class SynthDataset(Dataset):

    transform = transforms.Compose([transforms.ToTensor()])

    def __init__(self, csv_path, split=0.8):
        #os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.data:pd.DataFrame # not assigned
        self.csv_path = csv_path
        self.root_dir = os.path.dirname(self.csv_path)
        self.split = split

    @classmethod
    def from_file(cls, csv_path, split=0.8):
        self = cls(csv_path, split)
        self.data = pd.read_csv(csv_path)
        return self

    def get_train_test_dataloaders(self, dataset,  batch_size, shuffle, num_workers):
        train = SynthDataset(dataset.csv_path, dataset.split)
        train.data = dataset.data[:int(dataset.split * len(dataset.data))]
        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        test = SynthDataset(dataset.csv_path    , dataset.split)
        test.data = dataset.data[int(dataset.split * len(dataset.data)):]
        test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return (train_dataloader, test_dataloader)

    def __len__(self):
        return 10 #VAEmodel.latent_dimensions# len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        filename = str(self.data.iloc[index, 0])
        image_path = os.path.join(self.root_dir, filename)
        pixels = self.image_to_input(image_path)
        gen = self.gen_from_index(index)
        sample = {'filename': filename, 'image_path': image_path, 'image': pixels, 'gen':gen}
        # if self.transform:
        #     sample = self.transform(sample)
        return sample
    # def array_to_image(self, ar):
    #     pixels = np.array(pixels.tolist())[:,:,:3]
    #     pixels = pixels.reshape(32,32,3)/255
    #     pixels = np.array(pixels, 'float32')
    #     pixels = self.transform(pixels)

    def gen_from_index(self, index):
        gen = self.data.iloc[index, 1:]
        val = gen[0]
        ohe = [0,0,1] if val < .33 else ([0,1,0] if val < .66 else [[1,0,0]])

        gen = np.array(gen[1:])
        ohe = np.array(ohe)
        gen = np.concatenate([ohe, gen], axis=None)
        gen = gen.astype('float32')
        return gen
    
    @classmethod
    def image_to_input(cls, image_path):
        pixels = io.imread(image_path)
        pixels = np.array(pixels.tolist())[:,:,:3]
        pixels = pixels.reshape(32,32,3)/255
        pixels = np.array(pixels, 'float32')
        tensor:torch.Tensor = cls.transform(pixels)
        # DON'T PUT TENSORS ONTO CUDA IN DATASET
        #tensor = tensor.to(torch.device("cuda"))
        return tensor
    


    def getDrawParams(self, index):
        ar = self[index]
        filename = ar['filename']
        label_index = ar['label_index']
        gen = ar['gen']
        fCol = gen[0]
        sCol = gen[1]
        sW = gen[2]
        loc = Point(gen[3], gen[4])
        sza = SizeAspect(gen[5], gen[6])

        dp = DrawParams(filename, label_index, fCol, sCol, sW, loc, sza)
