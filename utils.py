import os
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
import torch
# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        super().__init__()
        image_paths = os.listdir(image_paths)
        #print( image_paths[0])
        image_paths = ["./data/FFHQ_128"+f"/{el}" for el in image_paths]
        self.image_paths = image_paths
        """self.transform = albumentations.Compose([
            albumentations.RandomCrop(height=128, width=128),
            albumentations.augmentations.transforms.HorizontalFlip(p=0.5)
        ])"""
        self.transform = torchvision.transforms.RandomHorizontalFlip()
        self.normalise = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        
        image = Image.open(self.image_paths[index])
        image = np.array(image)/255.0
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        image = self.transform(image)
        image = self.normalise(image)
        return image


class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    #train_data = ImagePaths(args.dataset_path, size=128)
    train_data = ImageDataset(args.dataset_path)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,num_workers = 8,
                              pin_memory = True)
    return train_loader


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()
