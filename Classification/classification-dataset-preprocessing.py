


import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


import pandas as pd
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation

class GlaucomaDataset(Dataset):
    def __init__(self, root_dir, split='train', output_size=(299,299), max_images=None):
        self.output_size = output_size
        self.root_dir = root_dir
        self.split = split
        self.images = []
        self.labels = []
        self.max_images = max_images

        # Define transformations
        self.transform = Compose([
            Resize(output_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization parameters from ImageNet
        ])

        if split == 'train':
            self.transform = Compose([
                Resize(output_size),
                RandomHorizontalFlip(),
                RandomRotation(20),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization parameters from ImageNet
            ])

        # Load labels
        self.labels_df = pd.read_csv(root_dir+"/"+root_dir + '.csv')

        self.image_filenames = []
        for path in os.listdir(os.path.join(root_dir, "Images_Square")):
            if(not path.startswith('.')):
                self.image_filenames.append(path)

        num_images = 0
        for k in range(len(self.image_filenames)):
            if max_images is not None and num_images >= max_images:
                break

            print('Loading {} image {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
            img_name = os.path.join(root_dir, "Images_Square", self.image_filenames[k])
            img = Image.open(img_name).convert('RGB')
            img = self.transform(img)
            self.images.append(img)
            
            # Get the label for this image from the DataFrame
            label = self.labels_df.loc[self.labels_df['imageID'] == self.image_filenames[k], 'binaryLabels'].values[0]
            self.labels.append(label)
            
            num_images += 1

        print('Successfully loaded {} dataset with {} images.'.format(split, len(self.images)) + ' '*50)

    def __len__(self):
        return len(self.images)
   
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert label to tensor if not already
        label = torch.tensor(label) if type(label) != torch.Tensor else label

        return img, label


# class GlaucomaDataset(Dataset):
#     def __init__(self, root_dir, split='train', output_size=(256,256), max_images=None):
#         self.output_size = output_size
#         self.root_dir = root_dir
#         self.split = split
#         self.images = []
#         self.labels = []
#         self.max_images = max_images

#         # Load labels
#         self.labels_df = pd.read_csv(root_dir+"/"+root_dir + '.csv')

#         self.image_filenames = []
#         for path in os.listdir(os.path.join(root_dir, "Images_Square")):
#             if(not path.startswith('.')):
#                 self.image_filenames.append(path)

#         num_images = 0
#         for k in range(len(self.image_filenames)):
#             if max_images is not None and num_images >= max_images:
#                 break

#             print('Loading {} image {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
#             img_name = os.path.join(root_dir, "Images_Square", self.image_filenames[k])
#             img = np.array(Image.open(img_name).convert('RGB'))
            
#             img = transforms.functional.to_tensor(img)
#             img = transforms.functional.resize(img, output_size, interpolation=Image.BILINEAR)
#             self.images.append(img)
            
#             # Get the label for this image from the DataFrame
#             label = self.labels_df.loc[self.labels_df['imageID'] == self.image_filenames[k], 'binaryLabels'].values[0]
#             self.labels.append(label)
            
#             num_images += 1

#         print('Succesfully loaded {} dataset with {} images.'.format(split, len(self.images)) + ' '*50)

#     def __len__(self):
#         return len(self.images)
   
#     def __getitem__(self, idx):
#         img = self.images[idx]
#         label = self.labels[idx]
        
#         # Convert label to tensor if not already
#         label = torch.tensor(label) if type(label) != torch.Tensor else label

#         return img, label

