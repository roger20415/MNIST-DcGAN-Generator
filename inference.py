import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from generator import Generator
from matplotlib import pyplot as plt
import numpy as np

from augmented_images import ImageAugmentationer
from config import Config

class Inferencer:
    def __init__(self) -> None:
        self.image_augmentationer = ImageAugmentationer()

    def show_inference_plot(self) -> None:
        if os.path.exists(Config.INFERENCE_PLOT_PATH):
            print(f"Inference plot already exists at {Config.INFERENCE_PLOT_PATH}")
            img = plt.imread(Config.INFERENCE_PLOT_PATH)
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        else:
            print("Inference plot does not exist. Running inference and saving the plot...")
            real_batch, fake_images = self.inference()
            self._save_inference_images(real_batch, fake_images)
            img = plt.imread(Config.INFERENCE_PLOT_PATH)
            plt.imshow(img)
            plt.axis("off")
            plt.show()
    
    def inference(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = Generator(Config.NGPU).to(device)
        
        try:
            generator.load_state_dict(torch.load(Config.NET_G_PATH, map_location=device, weights_only=True))
            generator.eval()
            print("Generator model loaded successfully.")
        except RuntimeError as e:
            print(f"Error loading the Generator model: {e}")
            return
        
        dataloader: DataLoader = self.image_augmentationer._load_mnist()
        real_batch = next(iter(dataloader))
        noise = torch.randn(64, 100, 1, 1, device=device)
        fake_images = generator(noise).detach().cpu()
        
        return real_batch, fake_images
    
    def _save_inference_images(self, real_batch: torch.Tensor, fake_images: torch.Tensor) -> None:
        plt.figure(figsize=(15, 15))

        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(make_grid(fake_images, padding=2, normalize=True), (1, 2, 0)))
        plt.savefig(Config.INFERENCE_PLOT_PATH)
        plt.close()
    
    
if __name__ == "__main__":
    inferencer = Inferencer()
    inferencer.show_inference_plot()
    