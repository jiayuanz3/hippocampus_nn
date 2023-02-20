import torch
import numpy as np
print('Torch Version: ',torch.__version__)

from src.dataloader import generate_data
from src.train import train

if __name__ == '__main__':
    file_path = 'Task04_Hippocampus/'
    traindataloader,valdataloader,testdataloader = generate_data(file_path)
    train(traindataloader, valdataloader)