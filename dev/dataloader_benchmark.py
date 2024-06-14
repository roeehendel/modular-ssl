import time
from argparse import ArgumentParser

from datamodules.legcay.wds_imagenet_datamodule import WDSImagenetDataModule


def main():
    device = 'cuda:0'

    batch_size = 64
    num_workers = 32

    parser = ArgumentParser()
    parser = WDSImagenetDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.batch_size = batch_size
    args.num_workers = num_workers
    datamodule = WDSImagenetDataModule(**vars(args))

    datamodule.prepare_data()
    datamodule.setup()

    dataloader = datamodule.train_dataloader()

    t0 = time.time()

    for i, (inputs, _) in enumerate(dataloader):
        print(f'Average time per batch ({i}): {(time.time() - t0) / (i + 1)}')


if __name__ == '__main__':
    main()
