import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def plot_example(file_path='Task04_Hippocampus/Task04_Hippocampus/imagesTr/hippocampus_003.nii'):
    test_load = nib.load(file_path).get_fdata()
    for i in range(5):
        plt.subplot(5, 5, i + 1)
        plt.imshow(test_load[:, :, 20 + i])
        plt.gcf().set_size_inches(10, 10)
    plt.show()