from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import lightning as L
import multiprocessing


class DiceDataModule(L.LightningDataModule):
    def __init__(self, data_dir='../data', batch_size=16, image_size=(64, 64)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_data = None
        self.val_data = None
        # self.transform = transforms.Compose([
        #   transforms.Grayscale(),
        #   transforms.Resize(image_size),
        #   transforms.RandomRotation(90),
        #   transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.1),
        #   transforms.RandomResizedCrop(self.image_size, scale=(0.9, 1.1)),
        #   transforms.ToTensor(),
        #   transforms.Normalize((0.5,), (0.5,))
        # ])
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_data, self.val_data = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=min(4, multiprocessing.cpu_count() - 1), persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                          num_workers=min(4, multiprocessing.cpu_count() - 1), persistent_workers=True)
