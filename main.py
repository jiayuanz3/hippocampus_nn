import torch
import numpy as np
print('Torch Version: ',torch.__version__)

from src.dataloader import generate_data,create_list
from src.train import train

if __name__ == '__main__':

    # Step 0: Download data from http://medicaldecathlon.com/

    # Step 1: Reshuffle the data into train, test and validation
    # Please add the corresponding function here

    # Step 2: Add the train, test data list here.
    '''
    file_path = 'Task04_Hippocampus/'
    create_list(file_path,data_type='train')
    create_list(file_path, data_type='val')
    create_list(file_path, data_type='test')
    '''


    # Step  3: Train!
    file_path = 'Task04_Hippocampus/'
    train_path = file_path + 'train/'
    val_path = file_path + 'val/'
    test_path = file_path + 'test/'
    traindataloader, valdataloader, testdataloader = generate_data(train_path, val_path, test_path)
    train(traindataloader, valdataloader)
    '''
    device = 'cpu'
    traindataloader, valdataloader, testdataloader = generate_data(file_path)
    from src.model import UNet
    model = UNet()
    model.load_state_dict(torch.load('UNet_hippocampus_best1676910724.pt', map_location=torch.device('cpu')))
    for i, (data, label) in enumerate(valdataloader):
        data, label = data.to(device), label.to(device)
        out = model(data)
        out_label = torch.argmax(out, dim=1)
        print(out_label.shape)
        print(torch.max(out_label))
        import matplotlib.pyplot as plt
        plt.hist(out_label.numpy().flatten())
        plt.show()

        raise ValueError('For try only!')
    '''



