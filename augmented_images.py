import os

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision.transforms import v2
from torchvision import datasets as dset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from config import Config

COMMON_SEED = 42

class ImageAugmentationer:

    def show_original_augmented_compare_plot(self) -> None:
        if not os.path.exists(Config.ORIGINAL_DATASET_PLOT_PATH):
            original_dataloader = self._load_mnist()
            self._save_mnist_grid_plot(original_dataloader, "Original", Config.ORIGINAL_DATASET_PLOT_PATH)

        if not os.path.exists(Config.AUGMENTED_DATASET_PLOT_PATH):
            augmented_dataloader = self._load_and_augment_mnist()
            self._save_mnist_grid_plot(augmented_dataloader, "Augmented", Config.AUGMENTED_DATASET_PLOT_PATH)
        
        self._combined_plots_and_show(Config.ORIGINAL_DATASET_PLOT_PATH, Config.AUGMENTED_DATASET_PLOT_PATH)

    def _load_mnist(self) -> DataLoader:
        self._set_random_seed(COMMON_SEED)
        transform = v2.Compose([
            v2.Resize(Config.NGF),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        dataset = dset.ImageFolder(root=Config.MNIST_FOLDER_PATH,
                                transform=transform)
        
        dataloader: DataLoader = torch.utils.data.DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                                                shuffle=True, num_workers=2)
        return dataloader

    def _load_and_augment_mnist(self) -> DataLoader:
        self._set_random_seed(COMMON_SEED)
        augmented_transform = v2.Compose([
            v2.Resize(Config.NGF),
            v2.RandomRotation(30),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        augmented_dataset = dset.ImageFolder(
            root=Config.MNIST_FOLDER_PATH, transform=augmented_transform
        )

        augmented_dataloader: DataLoader = torch.utils.data.DataLoader(
            augmented_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
        )
        return augmented_dataloader

    def _save_mnist_grid_plot(self, dataloader: DataLoader, title: str, save_path: str) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        augmented_batch = next(iter(dataloader))
        plt.figure(figsize=(64, 64))
        plt.axis("off")
        plt.title(title)
        plt.imshow(
            np.transpose(
                vutils.make_grid(augmented_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),
                (1, 2, 0),
            )
        )
        plt.savefig(save_path)

    def _set_random_seed(self, seed: int) -> None:
        # Let original plot and augmented plot have the same random seed, show the same numbers.
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    

    def _combined_plots_and_show(self, plot1_path: str, plot2_path: str) -> None:
        plot1 = mpimg.imread(plot1_path)
        plot2 = mpimg.imread(plot2_path)
        _, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(plot1)
        axes[0].axis("off")

        axes[1].imshow(plot2)
        axes[1].axis("off")
        plt.show()