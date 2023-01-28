
import os
import torch
import random
import numpy as np
import pandas as pd
# import openai
# import pickle
# import tiktoken
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import norm
from Utils import Utils
from InspectData import InspectData
from SkiaGen import SkiaGen
from VAE import VAEmodel
from SynthDataset import SynthDataset
from Train import TrainVAE, TrainDecoder
from Concept import Concept

batch_size = 30
inspect = InspectData()
Utils.SetHomeAsFileRoot()

def train(learning_rate):
    vaeTrain = TrainVAE(csv_path, batch_size, learning_rate)
    vaeTrain.train()

def continue_training(model_path:str, learning_rate):
    vaeTrain = TrainVAE(csv_path, batch_size, learning_rate)
    vaeTrain.resume(ckpt)
    vaeTrain.train()


def train_decoder(output_dir:str, learning_rate):
    decoderTrain = TrainDecoder(csv_path, batch_size, output_dir, learning_rate)
    decoderTrain.train()

def continue_decoder_training(model_path:str, learning_rate):
    output_dir = Utils.GetFolder(model_path)
    decoderTrain = TrainDecoder(csv_path, batch_size, output_dir, learning_rate)
    decoderTrain.resume(model_path)
    decoderTrain.train()

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
    show_latent_change_grid(vae, z_org)

def show_latent_change_decoder(index_list, ckpt_path):
    dataset = SynthDataset.from_file(csv_path)

    for i, index in enumerate(index_list):
        gen = dataset.gen_from_index(index)
        show_decoder_gen(gen, ckpt_path, index, True if len(index_list) - 1 else False)

def show_decoder_gen(gen, ckpt_path, index=0, show=True):
    gen = torch.from_numpy(gen)
    gen = gen.unsqueeze(0)
    model = VAEmodel.create_decoder_with_checkpoint(ckpt_path)
    imgs = show_latent_change_grid(model, gen)
    folder = Utils.GetFolder(ckpt_path)
    inspect.show_images(imgs, f'{folder}/_sep{index}.png', show)
    
def show_latent_change_grid(model, latent):
    count = 20
    imgs = list()
    mins = [.0,    0,0,0,  -1,-1,  0, -.8] #0,-1,
    maxs = [.6,    1,1,1,   1, 1, .6,  .8] #1, 1,
    for j_index in range(0, len(mins)):
        row = list()
        imgs.append(row)
        z = latent.clone()
        
        j = j_index - 3
        #val = float(z[0,j].cpu().detach().numpy())
        #j2 = j+1 if j < 6 else 0
        #val2 = float(z[0,j2].cpu().detach().numpy())
        for i in range(count):
            offset = ((maxs[j] - mins[j]) / float(count)) * i + mins[j]
            z[0,j_index] =  offset#val + (i-count//2) * .5
            #z[0,j2] =  val2 + (i-count//2) * .1
            img = model.sample_with_z(z)
            #img = model.sample(1)[0]
            img_pil:Image.Image = transforms.ToPILImage()(img)
            #if i == 4:
            row.append(img_pil)
    return imgs

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
    samples = gen.gen_data(folder_path, count, Concept.get_concepts())
    #inspect.show_images([samples], f'{folder_path}/_gen.png')

ckpt = './output/checkpoint_org.pt'
csv_path = './data/params.csv'

if __name__ == '__main__':  
    seed = 10
    np.random.seed(seed)
    random.seed(seed)
    #gen_data(8000, 0)
    #show_latents(44)
    

    #train(1e-3)
    #continue_training(ckpt,1e-3)
    #show_latent_change_full(706)
    
    decoder_path = "./output_concepts8"#output_full_ohe"
    #train_decoder(f"{decoder_path}", 1e-3)
    #continue_decoder_training(f'{decoder_path}/checkpoint_260.pt',1e-3)
    #show_latent_change_decoder(list(range(200,206)), f'{decoder_path}/checkpoint_260.pt')

    # ar = np.array([ 1.0000,  0.0000,  0.0000,  0.9534,  0.0039,  0.5122,  0.6252,  0.2251,   0.4887, -0.1249], 'float32')
    # show_decoder_gen(ar, f'{decoder_path}/checkpoint_140.pt')




















# import numpy as np
# import sklearn.decomposition
# import pickle
# import time


# # Apply 'Algorithm 1' to the ada-002 embeddings to make them isotropic, taken from the paper:
# # ALL-BUT-THE-TOP: SIMPLE AND EFFECTIVE POST- PROCESSING FOR WORD REPRESENTATIONS
# # Jiaqi Mu, Pramod Viswanath

# # This uses Principal Component Analysis (PCA) to 'evenly distribute' the embedding vectors (make them isotropic)
# # For more information on PCA, see https://jamesmccaffrey.wordpress.com/2021/07/16/computing-pca-using-numpy-without-scikit/


# # get the file pointer of the pickle containing the embeddings
# fp = open('/path/to/your/data/Embedding-Latest.pkl', 'rb')


# # the embedding data here is a dict consisting of key / value pairs
# # the key is the hash of the message (SHA3-256), the value is the embedding from ada-002 (array of dimension 1536)
# # the hash can be used to lookup the orignal text in a database
# E = pickle.load(fp) # load the data into memory

# # seperate the keys (hashes) and values (embeddings) into seperate vectors
# K = list(E.keys()) # vector of all the hash values 
# X = np.array(list(E.values())) # vector of all the embeddings, converted to numpy arrays


# # list the total number of embeddings
# # this can be truncated if there are too many embeddings to do PCA on
# print(f"Total number of embeddings: {len(X)}")

# # get dimension of embeddings, used later
# Dim = len(X[0])

# # flash out the first few embeddings
# print("First two embeddings are: ")
# print(X[0]) 
# print(f"First embedding length: {len(X[0])}")
# print(X[1])
# print(f"Second embedding length: {len(X[1])}")


# # compute the mean of all the embeddings, and flash the result
# mu = np.mean(X, axis=0) # same as mu in paper
# print(f"Mean embedding vector: {mu}")
# print(f"Mean embedding vector length: {len(mu)}")


# # subtract the mean vector from each embedding vector ... vectorized in numpy
# X_tilde = X - mu # same as v_tilde(w) in paper



# # do the heavy lifting of extracting the principal components
# # note that this is a function of the embeddings you currently have here, and this set may grow over time
# # therefore the PCA basis vectors may change over time, and your final isotropic embeddings may drift over time
# # but the drift should stabilize after you have extracted enough embedding data to characterize the nature of the embedding engine
# print(f"Performing PCA on the normalized embeddings ...")
# pca = sklearn.decomposition.PCA()  # new object
# TICK = time.time() # start timer
# pca.fit(X_tilde) # do the heavy lifting!
# TOCK = time.time() # end timer
# DELTA = TOCK - TICK

# print(f"PCA finished in {DELTA} seconds ...")

# # dimensional reduction stage (the only hyperparameter)
# # pick max dimension of PCA components to express embddings
# # in general this is some integer less than or equal to the dimension of your embeddings
# # it could be set as a high percentile, say 95th percentile of pca.explained_variance_ratio_
# # but just hardcoding a constant here
# D = 15 # hyperparameter on dimension (out of 1536 for ada-002), paper recommeds D = Dim/100


# # form the set of v_prime(w), which is the final embedding
# # this could be vectorized in numpy to speed it up, but coding it directly here in a double for-loop to avoid errors and to be transparent
# E_prime = dict() # output dict of the new embeddings
# N = len(X_tilde)
# N10 = round(N/10)
# U = pca.components_ # set of PCA basis vectors, sorted by most significant to least significant
# print(f"Shape of full set of PCA componenents {U.shape}")
# U = U[0:D,:] # take the top D dimensions (or take them all if D is the size of the embedding vector)
# print(f"Shape of downselected PCA componenents {U.shape}")
# for ii in range(N):
#     v_tilde = X_tilde[ii]
#     v = X[ii]
#     v_projection = np.zeros(Dim) # start to build the projection
#     # project the original embedding onto the PCA basis vectors, use only first D dimensions
#     for jj in range(D):
#         u_jj = U[jj,:] # vector
#         v_jj = np.dot(u_jj,v) # scaler
#         v_projection += v_jj*u_jj # vector
#     v_prime = v_tilde - v_projection # final embedding vector
#     v_prime = v_prime/np.linalg.norm(v_prime) # create unit vector
#     E_prime[K[ii]] = v_prime 

#     if (ii%N10 == 0) or (ii == N-1):
#         print(f"Finished with {ii+1} embeddings out of {N} ({round(100*ii/N)}% done)")


# # save as new pickle
# print("Saving new pickle ...")
# embeddingName = '/path/to/your/data/Embedding-Latest-Isotropic.pkl'
# with open(embeddingName, 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([E_prime,mu,U], f)
#     print(embeddingName)

# print("Done!")

# # When working with live data with a new embedding from ada-002, be sure to tranform it first with this function before comparing it
# #
# # def projectEmbedding(v,mu,U):
# #     v = np.array(v)
# #     v_tilde = v - mu
# #     v_projection = np.zeros(len(v)) # start to build the projection
# #     # project the original embedding onto the PCA basis vectors, use only first D dimensions
# #     for u in U:
# #         v_jj = np.dot(u,v) # scaler
# #         v_projection += v_jj*u # vector
# #     v_prime = v_tilde - v_projection # final embedding vector
# #     v_prime = v_prime/np.linalg.norm(v_prime) # create unit vector
# #     return v_prime 