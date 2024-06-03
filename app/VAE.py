import os, json, time, sys
from glob import glob
import numpy as np
from tqdm import tqdm
import cv2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.functional import F
from monai.transforms import RemoveRepeatedChannel
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
from model import VAE
#from generative.networks.nets import VQVAE

print_config()

image_size = 64
train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0,
                                        b_min=0.0, b_max=1.0, clip=True),
        transforms.RemoveRepeatedChanneld(keys=["image"], repeats=3),
    ]

)

val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0,
                                        b_min=0.0, b_max=1.0, clip=True),
        transforms.RemoveRepeatedChanneld(keys=["image"], repeats=3),
    ]
)

from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

world_size = torch.cuda.device_count()


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)


def load_train_validation_data():
    data_dir = '/workspace/data'
    filename = os.path.join(data_dir, "data_split_multigpu.json")

    fid = open(filename, 'r')
    data_dict = json.load(fid)
    fid.close()

    # num_train = int(len(data) * 0.8)
    # train_datalist = data[:num_train]
    # num_valid = int(len(data) * 0.1)
    # val_datalist = data[num_train:num_train + num_valid]
    # test = data[num_train + num_valid:]

    # data_dict = {'train': train_datalist, 'val': val_datalist, 'test': test}
    # fid = open(os.path.join(data_dir, 'data_split_multigpu.json'), 'w')
    # json.dump(data_dict, fid)
    # fid.close()

    train_datalist = data_dict['train']
    val_datalist = data_dict['val']

    train_ds = Dataset(data=train_datalist, transform=train_transforms)
    val_ds = Dataset(data=val_datalist, transform=val_transforms)
    """
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
        embedding_dim=32,
    )
    """
    #model = VanillaVAE(in_channels=3, latent_dim=256)
    model = VAE(imgChannels=1, zDim=256)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    return train_ds, val_ds, model, optimizer


def prepare_dataloader(train_ds, val_ds):
    train_loader = DataLoader(train_ds, batch_size=128, sampler=DistributedSampler(train_ds),
                              num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, sampler=DistributedSampler(val_ds),
                            num_workers=4)
    return train_loader, val_loader


def train(rank: int, world_size: int, save_every: int = 10,
          total_epochs: int = 100):
    ddp_setup(rank, world_size)
    print(rank, world_size)
    train_ds, val_ds, model, optimizer = load_train_validation_data()
    train_loader, val_loader = prepare_dataloader(train_ds, val_ds)
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in torch.load('/workspace/data/multi_gpu_model/vqvae_50.pt').items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    """
    # model.load_state_dict(torch.load('/workspace/data/multi_gpu_model/vqvae_50.pt'))
    model = model.to(rank)
    # load_model

    start_epoch = 0
    model = DDP(model, device_ids=[rank])
    out_dir = '/workspace/data/multi_gpu_model_VAE'
    if os.path.exists(out_dir) == 0:
        os.mkdir(out_dir)

    l1_loss = L1Loss()

    n_epochs = total_epochs
    val_interval = 1
    epoch_recon_loss_list = []
    epoch_quant_loss_list = []
    val_recon_epoch_loss_list = []
    intermediary_images = []
    n_example_images = 4
    val_loss = 10000

    total_start = time.time()

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(rank)
            optimizer.zero_grad(set_to_none=True)
            recon_images, mu, log_var = model(images)

            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = F.binary_cross_entropy(recon_images, images, reduction='sum') + kl_divergence

            loss.backward()
            optimizer.step()


    destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    save_every = 5
    total_epochs = 10
    mp.spawn(train, nprocs=world_size, args=(world_size, save_every, total_epochs))