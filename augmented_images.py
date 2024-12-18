import os

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose
from torchvision import datasets as dset
import matplotlib.pyplot as plt
import numpy as np

from config import Config


class ImageAugmentationShower:
    
    def load_mnist(self) -> DataLoader:
        dataset = dset.ImageFolder(root=Config.MNIST_FOLDER_PATH,
                                transform=transforms.Compose([
                                    transforms.Resize(28),
                                    transforms.CenterCrop(28),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        
        dataloader: DataLoader = torch.utils.data.DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                                                shuffle=True, num_workers=2)
        return dataloader
    
    def _save_original_mnist_plot(self, dataloader: DataLoader) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.savefig(Config.TRAINING_DATASET_PLOT_PATH)

    def augment_mnist(self) -> None:
        pass
    
    def _save_augmented_mnist_plot(self, dataloader: DataLoader) -> None:
        pass
    
    
if __name__ == '__main__':
    image_augmentation_shower = ImageAugmentationShower()
    dataloader = image_augmentation_shower.load_mnist()
    image_augmentation_shower._save_original_mnist_plot(dataloader)
    
    print("Done")