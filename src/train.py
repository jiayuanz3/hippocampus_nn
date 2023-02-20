import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .model import UNet
from .utils import *
def train(traindataloader,testdataloader):
    # call you model
    model = UNet(channel_in=1, channel_out=3)
    lr = 0.001
    # optimiser = optim.Adam(model.parameters(), lr=lr)
    # Optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # optimiser = torch.optim.RMSprop(model.parameters(), lr = lr, weight_decay = 1e-8, momentum=0.9)
    # TODO: you can try with different loss function
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    # Training & validation: same as your classification task!!
    model.to(device)
    model.train()
    # Tensorboard
    
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    # define no. of epochs you want to loop
    epochs = 10
    log_interval = 7  # for visualising your iterations

    # New: savining your model depending on your best val score
    best_valid_loss = float('inf')
    ckptFileName = 'UNet_hippocampus_best.pt'
    for epoch in range(epochs):
        train_loss, valid_loss, train_dsc, val_dsc = [], [], [], []

        for batch_idx, (data, label) in enumerate(traindataloader):
            # initialise all your gradients to zer

            label = label.to(device)
            optimiser.zero_grad()
            out = model(data.to(device))

            loss = criterion(out, label)
            loss.backward()
            optimiser.step()

            # append
            train_loss.append(loss.item())
            acc_1 = get_dice_arr(out, label.to(device))
            train_dsc.append(acc_1.mean(axis=0).detach().cpu().numpy())

            if (batch_idx % log_interval) == 0:
                print('Train Epoch is: {}, train loss is: {:.6f} and train dice: {:.6f}'.format(epoch,
                                                                                                np.mean(train_loss),
                                                                                                np.mean(train_dsc)))

                with torch.no_grad():
                    ### We add model.eval() during evalation.
                    model.eval()
                    for i, (data, label) in enumerate(valdataloader):
                        data, label = data.to(device), label.to(device)
                        out = model(data)
                        loss = criterion(out, label.to(device))
                        # acc_1 = get_dice_arr(out, label.to(device))
                        acc_1 = get_accuracy(out, label.to(device))

                        # append
                        val_dsc.append(acc_1.mean(axis=0).detach().cpu().numpy())
                        valid_loss.append(loss.item())

                print('Val Epoch is: {}, val loss is: {:.6f} and val dice: {:.6f}'.format(epoch, np.mean(valid_loss),
                                                                                          np.mean(val_dsc)))

        # Uncomment it to save your epochs
        if np.mean(valid_loss) < best_valid_loss:
            best_valid_loss = np.mean(valid_loss)
            print('saving my model, improvement in validation loss achieved...')
            torch.save(model.state_dict(), ckptFileName)

        # every epoch write the loss and accuracy (these you can see plots on tensorboard)
        writer.add_scalar('UNet/train_loss', np.mean(train_loss), epoch)
        writer.add_scalar('UNet/train_accuracy', np.mean(train_dsc), epoch)

        # New --> add plot for your val loss and val accuracy
        writer.add_scalar('UNet/val_loss', np.mean(valid_loss), epoch)
        writer.add_scalar('UNet/val_accuracy', np.mean(val_dsc), epoch)

    writer.close()