import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from matplotlib import pyplot as plt

from generator import Generator
from discriminator import Discriminator
from config import Config
from weight_initer import WeightIniter
from augmented_images import ImageAugmentationer

class Trainer:
    def __init__(self) -> None:
        self.image_augmentation_shower = ImageAugmentationer()
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and Config.NGPU > 0) else "cpu")
        
        self.netG = Generator(Config.NGPU).to(self.device)
        if (self.device.type == 'cuda') and (Config.NGPU > 1):
            self.netG = nn.DataParallel(self.netG, list(range(Config.NGPU)))
        self.netG.apply(WeightIniter.weights_init)
            
        self.netD = Discriminator(Config.NGPU).to(self.device)
        if (self.device.type == 'cuda') and (Config.NGPU > 1):
            self.netD = nn.DataParallel(self.netD, list(range(Config.NGPU)))
        self.netD.apply(WeightIniter.weights_init)
    
        
        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=Config.LR, betas=(Config.BETA1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=Config.LR, betas=(Config.BETA1, 0.999))
        self.fixed_noise = torch.randn(64, Config.NZ, 1, 1, device=self.device)
        self.real_label = 1
        self.fake_label = 0
    
    def show_netG_structure(self) -> None:
        print(self.netG)
    
    def show_netD_structure(self) -> None:
        print(self.netD)

    def show_loss_plot(self) -> None:
        if os.path.exists(Config.LOSS_PLOT_PATH):
            img = plt.imread(Config.LOSS_PLOT_PATH)
            plt.figure(figsize=(10, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        else:
            print("Loss plot does not exist. Train the model first.")
            self._training()
        
        
    def _training(self) -> None:
        # Training Loop

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        augmented_dataloader: DataLoader = self.image_augmentation_shower._load_and_augment_mnist()
        
        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(Config.EPOCHS):
            # For each batch in the dataloader
            for i, data in enumerate(augmented_dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, Config.NZ, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(self.fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, Config.EPOCHS, i, len(augmented_dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == Config.EPOCHS-1) and (i == len(augmented_dataloader)-1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
                
                self._save_loss_plot(G_losses, D_losses)
                
        torch.save(self.netG.state_dict(), Config.NET_G_PATH)
        torch.save(self.netD.state_dict(), Config.NET_D_PATH)
                
    def _save_loss_plot(self, G_losses: list, D_losses: list) -> None:
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(Config.LOSS_PLOT_PATH)