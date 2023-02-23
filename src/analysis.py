import glob
import nibabel as nib
import torch
import torchio as tio
from .model import UNet
def compute_test(model_path, subject_list,input_file,save_file_path):
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for i in range(len(subject_list)):
        image = tio.ScalarImage(glob.glob(input_file[i])[0])
        transform = tio.transforms.ZNormalization()
        image = torch.unsqueeze(torch.unsqueeze(transform(image).data.float(),dim=0),dim=0)
        output = torch.argmax(model(image), dim=1)[0].detach().numpy()

        image_obj = nib.load(input_file[i])
        ni_img = nib.Nifti1Image(output, image_obj.affine, image_obj.header)
        nib.save(ni_img, save_file_path + subject_list[i] + '_prediction.nii.gz')


