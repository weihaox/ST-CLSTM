import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
import json
import os
from dataloader.nyu_transform import *

try:
    import accimage
except ImportError:
    accimage = None


def load_annotation_data(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


def pil_loader(path):
    return Image.open(path)


def video_loader(root_dir, frame_indices):
    video = []
    for index in frame_indices:
        image_path = os.path.join(root_dir, index)
        if os.path.exists(image_path):
            video.append(pil_loader(image_path))
        else:
            return video

    return video


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dict_dir, root_dir, transform=None, is_test=False):
        self.data_dict = load_annotation_data(dict_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        rgb_index = self.data_dict[idx]['rgb_index']
        depth_index = self.data_dict[idx]['depth_index']
        test_index = self.data_dict[idx]['test_index']

        rgb_clips = video_loader(self.root_dir, rgb_index)
        depth_clips = video_loader(self.root_dir, depth_index)

        rgb_tensor = []
        depth_tensor = []
        depth_scaled_tensor = []
        for rgb_clip, depth_clip in zip(rgb_clips, depth_clips):
            sample = {'image': rgb_clip, 'depth': depth_clip}
            sample_new = self.transform(sample)
            rgb_tensor.append(sample_new['image'])
            depth_tensor.append(sample_new['depth'])

        return torch.stack(rgb_tensor, 0).permute(1, 0, 2, 3), \
               torch.stack(depth_tensor, 0).permute(1, 0, 2, 3), \
               test_index

    def __len__(self):
        return len(self.data_dict)

def getTrainingData_NYUDV2(batch_size=64, dict_dir=None, root_dir=None):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    transformed_training = depthDataset(dict_dir=dict_dir,
                                        root_dir = root_dir,
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [152, 114]),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'], 
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size, shuffle=True, num_workers=1, pin_memory=False)

    return dataloader_training

def getTestingData_NYUDV2(batch_size=64, dict_dir=None, root_dir=None, num_workers=4):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}


    transformed_testing = depthDataset(dict_dir=dict_dir,
                                       root_dir=root_dir,
                                       transform=transforms.Compose([
                                            Scale(240),
                                            CenterCrop([304, 228], [152, 114]),
                                            ToTensor(is_test=True),
                                            Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return dataloader_testing
