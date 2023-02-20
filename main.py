import torch
import numpy as np
print('Torch Version: ',torch.__version__)

from src.dataloader import generate_data
from src.train import train

if __name__ == '__main__':
    file_path = 'Task04_Hippocampus/'
    train_path = file_path + 'train/'
    val_path = file_path + 'val/'
    test_path = file_path + 'test/'
    traindataloader,valdataloader,testdataloader = generate_data(train_path,val_path,test_path)
    train(traindataloader, valdataloader)