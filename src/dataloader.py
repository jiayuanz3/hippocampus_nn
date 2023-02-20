import glob
import os

import numpy as np

import torch
import torchio as tio
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Load data and include prepared transform (Remember to apply same transform to both image and label) 
class mySegmentationData(object):
    def __init__(self, root, resize_shape, transforms = None):
        self.root = root
        self._eval = eval
        self.resize_shape = resize_shape
        self.transforms = transforms
        self.build_dataset()
                      
    def build_dataset(self):
        image_path = os.path.join(self.root, 'imagesTr')
        label_path = os.path.join(self.root, 'labelsTr')
        self._images = glob.glob(image_path + '/*.nii.gz')
        self._labels = glob.glob(label_path + '/*.nii.gz')
    
    def __getitem__(self, idx):
        image = tio.ScalarImage(self._images[idx])
        label = tio.ScalarImage(self._labels[idx])
        
        # normalization
        transform = tio.transforms.ZNormalization()
        image = transform(image)
        
        # resize
        transform = tio.CropOrPad(self.resize_shape)
        image = transform(image).data
        label = transform(label).data
        
        if self.transforms is not None:
            image = self.transforms(image)
            #label = self.transforms(label)

        label = torch.squeeze(label).type(torch.LongTensor)
   
        return image, label
    
    def __len__(self):
        return len(self._images)
    
def generate_data(file_path,resize_shape=(43,59,47),batch_size=32):
    dataset = mySegmentationData(file_path, resize_shape)
    num_sample = len(dataset)
    index = np.arange(0, num_sample)

    num_valid = int(np.floor(0.1 * num_sample))
    num_test = int(np.floor(0.1 * num_sample))


    valid_idx = np.random.choice(index, num_valid)
    index = index[~np.isin(index, valid_idx)]
    test_idx = np.random.choice(index, num_test)
    index = index[~np.isin(index, test_idx)]


    train_sampler = SubsetRandomSampler(index)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)


    # Now create data loaders (same as before)
    # Now we need to create dataLoaders that will allow to iterate during training
    #batch_size = 4 # create batch-based on how much memory you have and your data size

    traindataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    valdataloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,
                num_workers=0)
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler,
                num_workers=0)
    return traindataloader,valdataloader,testloader
    