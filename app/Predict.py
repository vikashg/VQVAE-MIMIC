import os, json, time, sys
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
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),]
        )

val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
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

model = VQVAE(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    num_channels=(256, 256),
    num_res_channels=256,
    num_res_layers=2,
    downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
    upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    num_embeddings=256,
    embedding_dim=1,
)

from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in torch.load('/workspace/data/multi_gpu_model/vqvae_100.pt').items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

device = torch.device("cuda:0")
model = model.to(device)

test_ds = Dataset(data=test_data, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

i = 0
for _test in test_loader:
    test = _test
    reconstruction, quant_loss = model(test['image'].to(device))
    x = model.encode(test['image'].to(device))
    for i in range(32):
        _test = x[0, i, :, :].cpu().detach().numpy()
        cv2.imwrite('/workspace/data/multi_gpu_model/test_' + str(i) + '.png', _test * 255.0)

    break

