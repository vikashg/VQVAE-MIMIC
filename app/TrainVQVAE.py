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


def main():
    data_dir = '/workspace/data'
    filename = os.path.join(data_dir, "data.json")
    #dist.init_process_group(backend="nccl", init_method="env://")
    fid = open(filename, 'r')
    data = json.load(fid)['Data']
    fid.close()

    num_train = int(len(data) * 0.8)
    train_datalist = data[:num_train]
    num_valid = int(len(data) * 0.1)
    val_datalist = data[num_train:num_train+num_valid]
    test = data[num_train+num_valid:]

    data_dict = {'train': train_datalist, 'val': val_datalist, 'test': test}
    fid = open(os.path.join(data_dir, 'data_split.json'), 'w')
    json.dump(data_dict, fid)
    fid.close()


    train_ds = Dataset(data=train_datalist, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)

    val_ds = Dataset(data=val_datalist, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    device = torch.device("cuda:0")
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
    ).to(device)
    #ddp_model = DDP(_model, device_ids=[device])
    #model = ddp_model

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    l1_loss = L1Loss()

    n_epochs = 100
    val_interval = 1
    epoch_recon_loss_list = []
    epoch_quant_loss_list = []
    val_recon_epoch_loss_list = []
    intermediary_images = []
    n_example_images = 4
    val_loss = 10000

    total_start = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)

            # model outputs reconstruction and the quantization error
            reconstruction, quantization_loss = model(images=images)

            recons_loss = l1_loss(reconstruction.float(), images.float())

            loss = recons_loss + quantization_loss

            loss.backward()
            optimizer.step()

            epoch_loss += recons_loss.item()

            progress_bar.set_postfix(
                {"recons_loss": epoch_loss / (step + 1), "quantization_loss": quantization_loss.item() / (step + 1)}
            )
        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_quant_loss_list.append(quantization_loss.item() / (step + 1))



        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(device)

                    reconstruction, quantization_loss = model(images=images)

                    # get the first sample from the first validation batch for
                    # visualizing how the training evolves
                    if val_step == 1:
                        intermediary_images.append(reconstruction[:n_example_images, 0])

                    recons_loss = l1_loss(reconstruction.float(), images.float())

                    val_loss += recons_loss.item()

            val_loss /= val_step
            val_recon_epoch_loss_list.append(val_loss)

        if (epoch + 1) % 25 == 0:
            # save model
            torch.save(model.state_dict(), os.path.join(data_dir, f"vqvae_{epoch+1}.pt"))

    total_time = time.time() - total_start
    #save the trained model
    torch.save(model.state_dict(), os.path.join(data_dir, 'vqvae.pt'))

    train_datalist = data[:num_train]
    a = train_transforms(train_datalist[0])
    print(model.encoder(a['image'].unsqueeze(0).to(device)).shape)

    plt.style.use("ggplot")
    plt.title("Learning Curves", fontsize=20)
    plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_recon_loss_list, color="C0", linewidth=2.0, label="Train")
    plt.plot(
        np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
        val_recon_epoch_loss_list,
        color="C1",
        linewidth=2.0,
        label="Validation",
    )
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(os.path.join(data_dir, 'learning_curve.png'))

if __name__ == '__main__':
    main()