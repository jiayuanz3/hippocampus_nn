import torch
import numpy as np
print('Torch Version: ',torch.__version__)

from src.dataloader import generate_data,create_list
from src.train import train

if __name__ == '__main__':
    weight_1_0 = np.linspace(1.0,5.0,5)
    weight_2_1 = np.linspace(0.5,1.5,5)
    for i in range(len(weight_1_0)):
        for j in range(len(weight_2_1)):
            config = {'channel_in': 1,  # Input channel = 1 since we're working on grey scale data
                      'channel_out': 3,  # Output channel = 3 since we have three classes
                      'lr': 0.001,  # Learning rate = 0.001
                      'epoch': 50,
                      'batch_size': 32,  # Batch size of training
                      'log_interval': 5,  # Log interval
                      'weight_decay': 1e-8,  # Weight decay for Adam
                      'loss_weight': [1., weight_1_0[i], weight_2_1[j] * weight_1_0[i]],
                      # Weight loss for different classes
                      'metric': 'accuracy',  # Validation metrics
                      }

            file_path = 'Task04_Hippocampus/'
            train_path = file_path + 'train/'
            val_path = file_path + 'val/'
            test_path = file_path + 'test/'
            save_root_path = 'result/'
            traindataloader, valdataloader, testdataloader = generate_data(train_path, val_path, test_path, config)
            train(traindataloader, valdataloader, config, save_root_path)
            print('weight_1_0=' + str(weight_1_0[i]) + 'training has completed!')