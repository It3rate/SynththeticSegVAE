
import torch
from torchvision import transforms
from PIL import Image
from VAE import VAEmodel
from Utils import Utils
from SynthDataset import SynthDataset

class TrainVAE:
    def __init__(self, csv_path:str, batch_size:int, model_path:str = ""):
        #os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.epochs = 10000
        self.output_dir = "./output/"
        Utils.EnsureFolder(self.output_dir)
        self.device = torch.device("cuda")
        self.dataset = SynthDataset.from_file(csv_path)
        self.train_dataloader, self.test_dataloader = \
            self.dataset.get_train_test_dataloaders(self.dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.model_path = model_path
        
        self.loss:torch.Tensor
        self.start_epoch = 0
        self.VAE = VAEmodel(latent_dims=VAEmodel.latent_dimensions, hidden_dims=[32, 64, 64], image_shape=[3,32,32])
        self.VAE = self.VAE.to(self.device)
        self.optimizer = torch.optim.Adam(self.VAE.parameters(), lr=1e-3)
        
    def resume(self, model_path:str):
        self.model_path = model_path
        checkpoint = torch.load(self.model_path)
        self.VAE.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

    def train(self):
        self.VAE.train()
        
        # recons:list = []
        # batch:dict = {}
        for epoch in range(self.start_epoch, self.epochs):
            for batch in self.train_dataloader:
                img = batch['image'].to(self.device)
                gen = batch['gen'].to(self.device)
                label_index = batch['label_index'].to(self.device)
                self.optimizer.zero_grad()
                recons, input, mu, log_var, _ = self.VAE.forward(img, gen)
                self.loss = self.VAE.loss_function(recons, input, mu, log_var)
                self.loss.backward()
                self.optimizer.step()

            print(f"Epoch: {epoch+1}         Loss: {self.loss:.4f}")
            if (epoch+1)%10==0:
                org_img = Image.open(batch["image_path"][0]) 
                org_img.save(f"{self.output_dir}/org_{epoch+1}.png")
                img_pil:Image.Image = transforms.ToPILImage()(recons[0])
                img_pil.save(f"{self.output_dir}/gen_{epoch+1}.png")
                #print(recons.shape)
                self.model_path = f'{self.output_dir}/checkpoint_{epoch+1}.pt'
                torch.save({
                    'model_state_dict': self.VAE.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': self.loss
                }, self.model_path)


class TrainDecoder:
    def __init__(self, csv_path:str, batch_size:int, output_dir:str, model_path:str = ""):
        #os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.epochs = 1000
        self.output_dir = Utils.GetFolder(output_dir)
        Utils.EnsureFolder(self.output_dir)
        self.device = torch.device("cuda")
        self.dataset = SynthDataset.from_file(csv_path)
        self.train_dataloader, self.test_dataloader = \
            self.dataset.get_train_test_dataloaders(self.dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.model_path = model_path
        
        self.loss:torch.Tensor
        self.start_epoch = 0
        self.model = VAEmodel(latent_dims=VAEmodel.latent_dimensions, hidden_dims=[32, 64, 64], image_shape=[3,32,32], create_model=False)
        self.model.create_decoder()
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
    def resume(self, model_path:str):
        self.model_path = model_path
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

    def train(self):
        self.model.train()
        
        # recons:list = []
        # batch:dict = {}
        for epoch in range(self.start_epoch, self.epochs):
            for batch in self.train_dataloader:
                img = batch['image'].to(self.device)
                gen = batch['gen'].to(self.device)
                self.optimizer.zero_grad()
                recons = self.model.forward_decode(gen)
                self.loss = self.model.loss_decode(recons, img)
                self.loss.backward()
                self.optimizer.step()

            print(f"Epoch: {epoch+1}         Loss: {self.loss:.4f}")
            if (epoch+1)%2==0:
                org_img = Image.open(batch["image_path"][0]) 
                org_img.save(f"{self.output_dir}/org_{epoch+1}.png")
                img_pil:Image.Image = transforms.ToPILImage()(recons[0])
                img_pil.save(f"{self.output_dir}/gen_{epoch+1}.png")
                #print(recons.shape)
                if (epoch+1)%20==0:
                    self.model_path = f'{self.output_dir}/checkpoint_{epoch+1}.pt'
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': self.loss
                    }, self.model_path)