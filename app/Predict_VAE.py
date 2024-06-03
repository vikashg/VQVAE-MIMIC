import os, json, time, sys
import tqdm
from glob import glob
import numpy as np
from tqdm import tqdm
import cv2
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import shutil
import tempfile
import time
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.nn import L1Loss
from tqdm import tqdm

from generative.networks.nets import VQVAE

print_config()

image_size = 64

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"], image_only=True),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0,
                                        b_min=0.0, b_max=1.0, clip=True),
        transforms.RemoveRepeatedChanneld(keys=["image"], repeats=3),
    ]

)

val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"], image_only=True),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0,
                                        b_min=0.0, b_max=1.0, clip=True),
        transforms.RemoveRepeatedChanneld(keys=["image"], repeats=3),
        transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224)),
    ]
)

from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

data_dir = '/workspace/data'
filename = os.path.join(data_dir, "data_split_multigpu.json")

fid = open(filename, 'r')
data_dict = json.load(fid)
fid.close()

test_data = data_dict['test']
train_data = data_dict['train']
print("test_data: ", len(test_data))
print("train_data: ", len(train_data))

#test_data = [{'image': './non-lung_img-10030487_52922406.jpg'}]
from model import VAE

from collections import OrderedDict


def predict_testset(model_fn, output_file, latent_dim=10):
    model = VAE(latent_dim=latent_dim)
    new_state_dict = OrderedDict()

    for k, v in torch.load(model_fn).items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    device = torch.device("cuda:0")
    model = model.to(device)

    test_ds = Dataset(data=test_data, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    i = 0
    result_mu_logvar = np.zeros((len(test_loader), latent_dim, 2))
    # columns = ['mu', 'logvar']

    for _test in tqdm(test_loader):
        test = _test

        #reconstruction, mu, logvar = model(test['image'].to(device))
        z = model.encoder(test['image'].to(device))
        print(z.shape)
        # print("MU ", mu.cpu().detach().numpy())
        # print("LOGVAR ", logvar.cpu().detach().numpy())
        #_mu = mu.cpu().detach().numpy()
        #_var = logvar.cpu().detach().numpy()
        #print(_mu, _var)
        #im = reconstruction.cpu().detach().numpy()
        #im = np.squeeze(im)
        #im = np.transpose(im, (1, 2, 0))
        #im = im * 255
        #im = im.astype(np.uint8)
        #print(im.shape)
        #cv2.imwrite("/workspace/data/multi_gpu_model_VAE_latent_size_10/results/{}.png".format(i), im)
        #result_mu_logvar[i, :, 0] = _mu
        #result_mu_logvar[i, :, 1] = _var
        i += 1

    #np.save(output_file, result_mu_logvar)


if __name__ == "__main__":
    #for i in tqdm(range(5, 205, 5)):
    i=200
    latent_dim = 10
    model_fn = "/workspace/data/multi_gpu_model_VAE_latent_size_10/models/vqvae_{}.pt".format(i)
    output_file = "/workspace/data/multi_gpu_model_VAE/results/result_mu_var{}.npy".format(i)
    predict_testset(model_fn, output_file, latent_dim=latent_dim)
