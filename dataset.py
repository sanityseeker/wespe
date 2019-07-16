import os

import torch
import torchvision
from torch.utils import data
from skimage import io

PATHTOPHONETEMPLATE = "/home/sanityseeker/Documents/dped/{0}/training_data/{0}/"
PATHTODSLRTEMPLATE = "/home/sanityseeker/Documents/dped/{0}/training_data/canon/"


class TrainingDataset(data.Dataset):
    def __init__(self, phone, file_names):
        self.phone = phone
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        # Select sample
        file_name = self.file_names[index]

        # Load data and get label
        path_to_phone_dir = PATHTOPHONETEMPLATE.format(self.phone)
        path_to_phone_file = os.path.join(path_to_phone_dir, file_name)
        path_to_dslr_dir = PATHTODSLRTEMPLATE.format(self.phone)
        path_to_dslr_file = os.path.join(path_to_dslr_dir, file_name)

        phone_image = io.imread(path_to_phone_file)
        dslr_image = io.imread(path_to_dslr_file)

        tensor_transform = torchvision.transforms.ToTensor()

        phone_image = tensor_transform(phone_image)

        dslr_image = tensor_transform(dslr_image)

        return phone_image, dslr_image


def to_grayscale(batch_tensor):
    rd = 0.299
    gr = 0.587
    bl = 0.114
    gray_batch = (rd * batch_tensor[:, 0, :].float() +
                  gr * batch_tensor[:, 1, :].float() +
                  bl * batch_tensor[:, 2, :].float())
    return gray_batch.unsqueeze(1)
