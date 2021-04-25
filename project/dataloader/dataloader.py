import os
from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST
from typing import Optional
from splitter.splitter import Splitter
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # download only
        FashionMNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        FashionMNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage: Optional[str] = None):
        # transform
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = FashionMNIST(os.getcwd(), train=True, download=False, transform=transform)
        mnist_test = FashionMNIST(os.getcwd(), train=False, download=False, transform=transform)


        # train/val split
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
