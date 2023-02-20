import torch
import numpy as np
print('Torch Version: ',torch.__version__)

from src.dataloader import generate_data
from src.train import train

if __name__ == '__main__':
    file_path = 'Task04_Hippocampus/'
    traindataloader, valdataloader, testdataloader = generate_data(file_path)
    #train(traindataloader, valdataloader)

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
