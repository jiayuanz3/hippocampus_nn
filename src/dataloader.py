import glob
import os

import numpy as np

import torch
import torchio as tio
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
r = np.random.RandomState(42)

def write_list_to_file(file_name,list_name):
    with open(file_name, 'w') as f:
        for line in list_name:
            f.write(f"{line}\n")

def read_file_to_list(file_name):
    list_name = []
    with open(file_name) as file:
        for line in file:
            list_name.append(line.rstrip())
    return list_name

def create_list(file_path,data_type):
    images_list = sorted(glob.glob(file_path + data_type + '/' + 'images/' + '/*.nii.gz'))
    labels_list = sorted(glob.glob(file_path + data_type + '/' + 'labels/' + '/*.nii.gz'))
    write_list_to_file(file_path + data_type + '/' + 'images_list.txt',images_list)
    write_list_to_file(file_path + data_type + '/' + 'labels_list.txt', labels_list)


# Load data and include prepared transform (Remember to apply same transform to both image and label) 
class mySegmentationData(object):
    def __init__(self, root):
        self.root = root
        self._eval = eval
        self.build_dataset()
                      
    def build_dataset(self):
        image_path = os.path.join(self.root, 'images')
        label_path = os.path.join(self.root, 'labels')
        self._images = sorted(glob.glob(image_path + '/*.nii.gz'))
        self._labels = sorted(glob.glob(label_path + '/*.nii.gz'))
    
    def __getitem__(self, idx):
        image = tio.ScalarImage(self._images[idx])
        label = tio.ScalarImage(self._labels[idx])

        return image.data, torch.squeeze(label.data.type(torch.LongTensor))
    
    def __len__(self):
        return len(self._images)
    
def generate_data(train_path, val_path, test_path, batch_size = 32):
    train = mySegmentationData(train_path)
    val = mySegmentationData(val_path)
    test = mySegmentationData(test_path)

    # Now create data loaders (same as before)
    # Now we need to create dataLoaders that will allow to iterate during training
     # create batch-based on how much memory you have and your data size

    traindataloader = DataLoader(train, batch_size=batch_size, num_workers=0)
    valdataloader = DataLoader(val, batch_size=batch_size, num_workers=0)
    testloader = DataLoader(test, batch_size=batch_size, num_workers=0)
    return traindataloader,valdataloader,testloader
    