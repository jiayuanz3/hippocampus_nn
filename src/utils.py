import numpy as np
import torch
from torchmetrics import Dice
# Create metrics

def mask_out(out):
    """Mask tensor/array with 0 threshold"""
    # Need to binarize the output to be able to calculate dice score
    return torch.argmax(out,dim=1)

def get_dice(out,label,device):
    dice = Dice(num_classes=3,ignore_index=0).to(device)
    return dice.forward(mask_out(out),label)

def get_accuracy(out,label):
    return torch.sum(mask_out(out)==label)/torch.numel(label)