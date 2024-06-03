import os, json, time, sys
from glob import glob
import numpy as np
from tqdm import tqdm
import cv2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
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

print_config()
from torch.nn import functional as F

def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='mean'   )
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


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

    train_datalist = data_dict['train']
    val_datalist = data_dict['val']
    print("train_datalist", len(train_datalist))
    print("val_datalist", len(val_datalist))

    train_ds = Dataset(data=train_datalist, transform=train_transforms)
    val_ds = Dataset(data=val_datalist, transform=val_transforms)

    model = VAE(latent_dim=10)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)

    return train_ds, val_ds, model, optimizer

def prepare_dataloader_multigpu(train_ds, val_ds):
    train_loader = DataLoader(train_ds, batch_size=128,  sampler=DistributedSampler(train_ds),
                              num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, sampler=DistributedSampler(val_ds),
                            num_workers=4)
    return train_loader, val_loader

def prepare_dataloader(train_ds, val_ds):
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=True,
                            num_workers=4)
    return train_loader, val_loader

def train(rank: int, world_size: int, save_every:int=10,
          total_epochs:int=100, start_epoch:int=0, val_interval:int=1,  ):

    out_dir = '/workspace/data/multi_gpu_model_VAE_latent_size_' + str(10) + '/'

    ddp_setup(rank, world_size)
    multi_gpu = 1
    train_ds, val_ds, model, optimizer = load_train_validation_data()

    step_lr = StepLR(optimizer, step_size=10, gamma=0.1)

    if multi_gpu == 1:
        device = rank
        train_loader, val_loader = prepare_dataloader_multigpu(train_ds, val_ds)
    else:
        device = torch.device(f"cuda:{rank}")
        train_loader, val_loader = prepare_dataloader(train_ds, val_ds)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    model_filename = os.path.join(out_dir, 'vqvae_' + str(start_epoch) + '.pt')
    if os.path.exists(model_filename):
        print("model found")
        for k, v in torch.load(model_filename).items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        print("model not found")
        if os.path.exists(out_dir) == 0:
            os.mkdir(out_dir)

    model = model.to(rank)
    # load_model

    if multi_gpu == 1:
        model = DDP(model, device_ids=[rank])
    else:
        model = model.to(device)
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
            # model outputs reconstruction and the quantization error
            reconstruction, mu, log_var = model(images)
            loss = loss_function(reconstruction, images, mu, log_var)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix(
                {"loss": loss.item() / (step + 1)}
            )
        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_quant_loss_list.append(loss.item() / (step + 1))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(rank)
                    reconstruction, mu, log_var = model(images)
                    # get the first sample from the first validation batch for
                    # visualizing how the training evolves
                    if val_step == 1:
                        intermediary_images.append(reconstruction[:n_example_images, 0])

                    recons_loss = loss_function(reconstruction, images, mu, log_var)
                    val_loss += recons_loss.item()

            val_loss /= val_step
            val_recon_epoch_loss_list.append(val_loss)

        if (epoch + 1) % save_every == 0:
            # save model
            torch.save(model.state_dict(), os.path.join(out_dir, f"vqvae_{epoch+1}.pt"))

        step_lr.step()
    total_time = time.time() - total_start
    #save the trained model
    torch.save(model.state_dict(), os.path.join(out_dir, f"vqvae_{epoch+1}.pt"))


    plt.style.use("ggplot")
    plt.title("Learning Curves", fontsize=20)
    plt.plot(np.linspace(start = start_epoch, stop = start_epoch + total_epochs,
                         num = total_epochs - start_epoch), epoch_recon_loss_list,
             color="C0", linewidth=2.0, label="Train")
    """
    plt.plot(
        np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
        val_recon_epoch_loss_list,
        color="C1",
        linewidth=2.0,
        label="Validation",
    )
    """
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    #plt.legend(prop={"size": 14})
    plt.savefig(os.path.join(out_dir, 'learning_curve2.png'))
    destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    save_every = 5
    total_epochs = 200
    mp.spawn(train, nprocs=world_size, args = (world_size, save_every, total_epochs, 100),
             )
