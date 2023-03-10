import torch
import numpy as np
print('Torch Version: ',torch.__version__)

from src.dataloader import generate_data,create_list
from src.train import train
from src.analysis import compute_test

if __name__ == '__main__':
    config = {'channel_in':1, # Input channel = 1 since we're working on grey scale data
              'channel_out':3,# Output channel = 3 since we have three classes
              'lr':0.001, # Learning rate = 0.001
              'epoch':50,
              'batch_size':32, # Batch size of training
              'log_interval':5, # Log interval
              'weight_decay':1e-8, # Weight decay for Adam
              'loss_weight':[1.,5.,2.5], # Weight loss for different classes
              'metric':'Dice', # Validation metrics: 'accuracy' or 'Dice'
              }
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

    '''
    # Step 3: Train!
    file_path = 'Task04_Hippocampus/'
    train_path = file_path + 'train/'
    val_path = file_path + 'val/'
    test_path = file_path + 'test/'
    save_root_path = 'result/'
    traindataloader, valdataloader, testdataloader = generate_data(train_path, val_path, test_path,config)
    train(traindataloader, valdataloader,config,save_root_path)
    '''


    # Step 4: compute the model for test
    model_file = 'result/20230222030004/UNet_hippocampus_best.pt'
    subject_list = ['100206','100307','100408']
    input_file = ['HCP_sample/'+ subject + '/MNINonLinear/T1w.nii.gz' for subject in subject_list]
    save_file_path = 'result/'
    compute_test(model_file, subject_list,input_file,save_file_path)
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



