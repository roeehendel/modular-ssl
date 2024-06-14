import sys
import time

import numpy as np
import torch
from pretrain import get_arg_parser, get_datamodule, get_encoder
from pytorch_lightning import seed_everything
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def batch_random_mask(batch_size, seq_len, num_keep):
    idx_keep = torch.argsort(torch.rand((batch_size, seq_len)), dim=-1)[:, :num_keep]
    return idx_keep


def time_per_batch(model, dataloader, device, percent_keep=1.0, num_batches=200):
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_counter = 0

    batch_times = []
    forward_times = []
    backward_times = []
    all_times = []

    samples_counter = 0

    t0_batch = time.time()
    t0_all = time.time()
    for i, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_time = time.time() - t0_batch
        batch_times.append(batch_time)
        torch.cuda.synchronize()

        t0_forward = time.time()

        inputs = inputs[0]

        samples_counter += inputs.shape[0]

        inputs = inputs[:128]

        inputs = inputs.to(device)

        if percent_keep < 1.0:
            idx_keep = batch_random_mask(inputs.shape[0], model.num_patches, int(model.num_patches * percent_keep))
            idx_keep = idx_keep.to(device)
            inputs = (inputs, idx_keep)

        # with autocast():
        #     out = model(inputs)
        #     loss = out.mean()

        torch.cuda.synchronize()
        forward_time = time.time() - t0_forward
        forward_times.append(forward_time)

        t0_backward = time.time()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        torch.cuda.synchronize()
        backward_time = time.time() - t0_backward
        backward_times.append(backward_time)

        all_time = time.time() - t0_all
        all_times.append(all_time)

        batch_counter += 1
        if batch_counter >= num_batches:
            break

        warmup_batches = 10
        if batch_counter % 10 == 0 and batch_counter > warmup_batches:
            print(samples_counter)
            print('Batch:', np.mean(batch_times[warmup_batches:]), np.std(batch_times[warmup_batches:]))
            print('Forward:', np.mean(forward_times[warmup_batches:]), np.std(forward_times[warmup_batches:]))
            print('Backward:', np.mean(backward_times[warmup_batches:]), np.std(backward_times[warmup_batches:]))
            print('All:', np.mean(all_times[warmup_batches:]), np.std(all_times[warmup_batches:]))

        t0_batch = time.time()
        t0_all = time.time()

    torch.cuda.synchronize()

    return np.mean(batch_times), np.std(batch_times)


def plot_batch_time_by_percent_keep(model, dataloader, device, num_batches):
    import matplotlib.pyplot as plt

    percent_keep_list = np.linspace(0.1, 1.0, 10)
    mean_time_per_batch_list = []
    std_time_per_batch_list = []

    for percent_keep in tqdm(percent_keep_list):
        model.patch_embed.patches_per_view = percent_keep
        mean_time, std_time = time_per_batch(model, dataloader, device, percent_keep, num_batches=num_batches)
        mean_time_per_batch_list.append(mean_time)
        std_time_per_batch_list.append(std_time)

    plt.plot(percent_keep_list, mean_time_per_batch_list)
    # plot error bars
    plt.fill_between(
        percent_keep_list,
        np.array(mean_time_per_batch_list) - np.array(std_time_per_batch_list),
        np.array(mean_time_per_batch_list) + np.array(std_time_per_batch_list),
        alpha=0.2
    )
    plt.show()


def main():
    print(sys.argv)

    sys.argv = [
        sys.argv[0],
        '--encoder_type=resnet',
        '--variant=18',
        '--dataset=imagenet',
        '--device_list=1',
        '--batch_size=512',
        '--num_workers=32',
    ]

    parser = get_arg_parser()
    args = parser.parse_args()

    device = f'cuda:0'

    num_batches = args.num_workers * 50

    print("arguments:")
    print(vars(args))

    seed_everything(args.random_seed, workers=True)

    encoder = get_encoder(args)
    datamodule = get_datamodule(args)

    datamodule.prepare_data()
    datamodule.setup()

    dataloader = datamodule.train_dataloader()

    print(len(datamodule.train_dataloader()), len(datamodule.val_dataloader()[0]), len(datamodule.val_dataloader()[1]))
    exit()

    encoder.train()
    encoder = encoder.to(device)

    print('performing test')

    time_per_batch(encoder, dataloader, device, percent_keep=1.0, num_batches=num_batches)

    # plot_batch_time_by_percent_keep(encoder, dataloader, device, num_batches)


if __name__ == '__main__':
    main()
