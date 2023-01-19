
import os
import torch
from InspectData import InspectData
from SkiaGen import SkiaGen
from VAE import VAEmodel
from SynthDataset import SynthDataset
from Train import TrainVAE, TrainDecoder
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

ckpt = './output/checkpoint_org.pt'
csv_path = './data/params.csv'
batch_size = 30
inspect = InspectData()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
    
    
def show_latent_change_grid():
    vae = VAEmodel.create_with_checkpoint(ckpt)
    index = 706
    input = SynthDataset.image_to_inputX(f'./data/img_{index}.png')
    mu, log_var = vae.encode(input.unsqueeze(0))
    count = 9
    imgs = list()#np.zeros(count * 7)
    z_org = vae.reparameterize(mu, log_var)
    for j in range(7):
        row = list()
        imgs.append(row)
        z = z_org.clone()
        val = z[0,j].cpu().detach().numpy()
        #j2 = j+1 if j < 6 else 0
        #val2 = z[0,j2].cpu().detach().numpy()
        for i in range(count):
            z[0,j] =  val + (i-count//2) * .6
            #z[0,j2] =  val2 + (i-count//2) * .6
            print(z.cpu().detach().numpy()[0])
            img = vae.sample_with_z(z)
            img_pil:Image.Image = transforms.ToPILImage()(img)
            #if i == 4:
            row.append(img_pil)

    inspect.show_images(imgs, f'./results/paramRange{index}.png')

def show_latents(index):
    vae = VAEmodel.create_with_checkpoint(ckpt)
    input = SynthDataset.image_to_inputX(f'./data/img_{index}.png')
    mu, log_var = vae.encode(input.unsqueeze(0))
    mu = mu.cpu().detach().numpy()[0]
    log_var = log_var.cpu().detach().numpy()[0]
    print(mu)
    print(log_var)
    # mu = [1,2,3,4,5,6,7]
    # log_var = [3,2,3,4,2,3,1]
    x = np.linspace(-2.5, 2.5, 1000)
    for mean, lv in zip(mu, log_var):
        std = np.exp(0.5*lv)
        y = norm.pdf(x, mean, std)
        plt.plot(x, y, label='m={:.2f}, lv={:.2f}'.format(mean, lv))

    plt.legend()
    plt.show()

def gen_data(count:int, start_index:int):
    samples = SkiaGen.gen_data(count, start_index)
    inspect.show_images([samples], f'./results/gen.png')

if __name__ == '__main__':  
    #show_latent_change_grid()
    #show_latents(44)
    # gen_data(50, 5000)
    #continue_training(ckpt)
    #train()
    train_decoder()
