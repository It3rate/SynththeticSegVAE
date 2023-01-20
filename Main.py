
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from Utils import Utils
from InspectData import InspectData
from SkiaGen import SkiaGen
from VAE import VAEmodel
from SynthDataset import SynthDataset
from Train import TrainVAE, TrainDecoder

ckpt = './output/checkpoint_org.pt'
csv_path = './data/params.csv'
batch_size = 30
inspect = InspectData()
Utils.SetHomeAsFileRoot()

def train():
    vaeTrain = TrainVAE(csv_path, batch_size)
    vaeTrain.train()

def train_decoder():
    decoderTrain = TrainDecoder(csv_path, batch_size)
    decoderTrain.train()

def continue_training(model_path:str):
    vaeTrain = TrainVAE(csv_path, batch_size)
    vaeTrain.resume(ckpt)
    vaeTrain.train()

def show_samples():
    vae = VAEmodel.create_with_checkpoint(ckpt)
    samples = vae.sample_images(10)
    inspect.show_images(samples)
    

def show_latent_change_full(index):
    path = f'./data/img_{index}.png'
    Utils.EnsureFolder(path)
    vae = VAEmodel.create_with_checkpoint(ckpt)
    input = SynthDataset.image_to_input(path)
    input = input.cuda()
    mu, log_var = vae.encode(input.unsqueeze(0))
    z_org = vae.reparameterize(mu, log_var)
    show_latent_change_grid(vae, z_org, index)

def show_latent_change_decoder(index):
    dataset = SynthDataset.from_file(csv_path)
    gen = dataset.gen_from_index(index)
    gen = torch.from_numpy(gen)
    gen = gen.unsqueeze(0)
    path = './output_zKnown/checkpoint_100.pt'
    model = VAEmodel.create_decoder_with_checkpoint(path)
    show_latent_change_grid(model, gen, index)
    
def show_latent_change_grid(model, latent, index:int):
    count = 20
    imgs = list()#np.zeros(count * 7)
    mins = [-.4, 0, .2, -6,-6, 0, .9]
    maxs = [1.0, 1, 4,  6,  6, 1, 2.5]
    for j in range(7):
        row = list()
        imgs.append(row)
        z = latent.clone()
        #val = float(z[0,j].cpu().detach().numpy())
        #j2 = j+1 if j < 6 else 0
        #val2 = float(z[0,j2].cpu().detach().numpy())
        for i in range(count):
            offset = ((maxs[j] - mins[j]) / float(count)) * i + mins[j]
            z[0,j] =  offset#val + (i-count//2) * .5
            #z[0,j2] =  val2 + (i-count//2) * .1
            print(i, z.cpu().detach().numpy()[0,2:])
            img = model.sample_with_z(z)
            img_pil:Image.Image = transforms.ToPILImage()(img)
            #if i == 4:
            row.append(img_pil)

    inspect.show_images(imgs, f'./results/sep_{index}.png')

def show_latents(index):
    vae = VAEmodel.create_with_checkpoint(ckpt)
    input = SynthDataset.image_to_input(f'./data/img_{index}.png')
    mu, log_var = vae.encode(input.unsqueeze(0))
    mu = mu.cpu().detach().numpy()[0]
    log_var = log_var.cpu().detach().numpy()[0]
    x = np.linspace(-2.5, 2.5, 1000)
    for mean, lv in zip(mu, log_var):
        std = np.exp(0.5*lv)
        y = norm.pdf(x, mean, std)
        plt.plot(x, y, label='m={:.2f}, lv={:.2f}'.format(mean, lv))

    plt.legend()
    plt.show()

def gen_data(count:int, start_index:int):
    folder_path = './data'
    gen = SkiaGen()
    samples = gen.gen_data(folder_path, count)
    #inspect.show_images([samples], f'{folder_path}/_gen.png')

if __name__ == '__main__':  
    #np.random.seed(10)
    #show_latent_change_full(706)
    #show_latent_change_decoder(458)
    #show_latents(44)
    #gen_data(5000, 0)
    #continue_training(ckpt)
    #train()
    train_decoder()
