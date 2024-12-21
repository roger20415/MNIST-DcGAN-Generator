import torch
from torch import nn, optim

from generator import Generator
from discriminator import Discriminator
from config import Config
from weight_initer import WeightIniter

class trainer:
    def __init__(self):
        device = torch.device("cuda:0" if (torch.cuda.is_available() and Config.NGPU > 0) else "cpu")
        
        self.netG = Generator(Config.NGPU).to(device)
        if (device.type == 'cuda') and (Config.NGPU > 1):
            self.netG = nn.DataParallel(self.netG, list(range(Config.NGPU)))
        self.netG.apply(WeightIniter.weights_init)
            
        self.netD = Discriminator(Config.NGPU).to(device)
        if (device.type == 'cuda') and (Config.NGPU > 1):
            self.netD = nn.DataParallel(self.netD, list(range(Config.NGPU)))
        self.netD.apply(WeightIniter.weights_init)
    
        
        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=Config.LR, betas=(Config.BETA1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=Config.LR, betas=(Config.BETA1, 0.999))
        self.fixed_noise = torch.randn(64, Config.NZ, 1, 1, device=device)
        self.real_label = 1
        self.fake_label = 0
    
    def show_netG_structure(self):
        print(self.netG)
    
    def show_netD_structure(self):
        print(self.netD)
        


if __name__ == "__main__":
    trainer = trainer()
    trainer.show_netG_structure()
    trainer.show_netD_structure()