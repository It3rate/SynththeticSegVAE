
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np

class VAEmodel(nn.Module):
    def __init__(self, latent_dims, hidden_dims, image_shape, create_model = True):
        super(VAEmodel, self).__init__()
        
        self.device = torch.device("cuda")
        self.latent_dims = latent_dims #Size of the latent space layer
        self.hidden_dims = hidden_dims #List of hidden layers number of filters/channels
        self.image_shape = image_shape #Input image shape

        if create_model:
            self.create_full_model()
        
    def create_full_model(self):
        self.last_channels = self.hidden_dims[-1]
        self.in_channels = self.image_shape[0]
        #Simple formula to get the number of neurons after the last convolution layer is flattened
        self.flattened_channels = int(self.last_channels*(self.image_shape[1]/(2**len(self.hidden_dims)))**2) 
       
        self.encoder = self.create_encoder()

        # Here are our layers for our latent space distribution
        self.fc_mu = nn.Linear(self.flattened_channels, self.latent_dims)
        self.fc_var = nn.Linear(self.flattened_channels, self.latent_dims)

        self.decoder = self.create_decoder()
        

    def create_encoder(self):
        # For each hidden layer we will create a Convolution Block
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels,
                              out_channels=h_dim,
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            self.in_channels = h_dim

        return nn.Sequential(*modules)

    def create_decoder(self):
        # Decoder input layer
        self.decoder_input = nn.Linear(self.latent_dims, self.flattened_channels)
        
        # For each Convolution Block created on the Encoder we will do a symmetric Decoder with the same Blocks, but using ConvTranspose
        rev_hidden_dims = self.hidden_dims.copy()
        rev_hidden_dims.reverse()
        modules = []
        for h_dim in rev_hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.in_channels,
                                       out_channels=h_dim,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            self.in_channels = h_dim

        # The final layer the reconstructed image have the same dimensions as the input image
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.image_shape[0],
                      kernel_size=3,
                      padding=1),
            nn.Sigmoid()
        )
        
        return nn.Sequential(*modules)
        

    @classmethod
    def create_with_checkpoint(cls, model_path:str):
        vae = VAEmodel(latent_dims=7, hidden_dims=[32, 64, 64], image_shape=[3,32,32])
        vae = vae.to(torch.device("cuda"))
        checkpoint = torch.load(model_path)
        vae.load_state_dict(checkpoint['model_state_dict'])
        return vae

    def get_latent_dims(self):
        return self.latent_dims
        
    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var componentsbof the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.last_channels, int(self.image_shape[1]/(2**len(self.hidden_dims))), int(self.image_shape[1]/(2**len(self.hidden_dims))))
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var, z]
    
    def loss_function(self, recons, input, mu, log_var):
        recons_loss = nn.functional.binary_cross_entropy(recons.reshape(recons.shape[0],-1),
                                                         input.reshape(input.shape[0],-1),
                                                         reduction="none").sum(dim=-1)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        loss = (recons_loss + kld_loss).mean(dim=0)
        return loss
        
    def generate(self, x):
        return self.forward(x)[0]
        
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dims)
        z = z.to(self.device)
        samples = self.decode(z)
        return samples

    def sample_with_z(self, z):
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        z = z.unsqueeze(0)
        z = z.to(self.device)
        z = z.detach() 
        sample = self.decode(z)
        return sample.cpu().detach()[0]
        
    def sample_images(self, num_samples):
        samples = self.sample(num_samples)
        samples = samples.cpu()
        result = list()
        for i in range(num_samples):
            result.append(transforms.ToPILImage()(samples[i]))
        return result
    
    def interpolate(self, starting_inputs, ending_inputs, granularity=10):
        """This function performs a linear interpolation in the latent space of the autoencoder
        from starting inputs to ending inputs. It returns the interpolation trajectories.
        """
        mu, log_var = self.encode(starting_inputs.to(self.device))
        starting_z = self.reparameterize(mu, log_var)
        
        mu, log_var = self.encode(ending_inputs.to(self.device))
        ending_z  = self.reparameterize(mu, log_var)
        
        t = torch.linspace(0, 1, granularity).to(self.device)
        
        intep_line = (
            
            torch.kron(starting_z.reshape(starting_z.shape[0], -1), (1 - t).unsqueeze(-1))+
            torch.kron(ending_z.reshape(ending_z.shape[0], -1), t.unsqueeze(-1))
            
        )
    
        decoded_line = self.decode(intep_line).reshape(
            (
                starting_inputs.shape[0],
                t.shape[0]
            )
            + (starting_inputs.shape[1:])
        )
        return decoded_line