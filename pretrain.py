import argparse
import os
from argparse import ArgumentParser
from tabnanny import verbose
from typing import Type

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger

import datamodules
import ssl_methods
from config import OUTPUTS_DIR
from datamodules.ssl_datamodule import SSLDataModule
from online_evaluators.knn_online_evaluator import KNNOnlineEvaluator
from online_evaluators.linear_online_evaluator import LinearProbeOnlineEvaluator
from online_evaluators.online_evaluation_runner import OnlineEvaluationRunner
from ssl_methods.ssl_method import SSLMethod


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    print("arguments:")
    print(vars(args))

    seed_everything(args.random_seed, workers=True)

    method = get_method(args)
    datamodule = get_datamodule(args)
    attach_transforms(datamodule, method)
    loggers = get_loggers(method, args)
    callbacks = get_callbacks(method, args)

    trainer = pl.Trainer.from_argparse_args(args, logger=loggers, callbacks=callbacks)
    trainer.fit(model=method, datamodule=datamodule)


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # Random seed
    parser.add_argument('--random_seed', type=int, default=42)

    # Dataset
    datasets_list = datamodules.registry.keys()
    parser.add_argument("--dataset", type=str, default="cifar10", help="Pretraining dataset", choices=datasets_list)
    parser.add_argument("--datasets_dir", type=str, default=os.environ.get('DATASETS-DIR'), help="Path of data dir")

    # SSL Method
    ssl_methods_list = ssl_methods.registry.keys()
    parser.add_argument("--method", type=str, default="simsiam", help="The SSL method to use", choices=ssl_methods_list)

    # Trainer
    parser = Trainer.add_argparse_args(parser)
    # TODO: check if it's better to use strategy='ddp_find_unused_parameters_false'
    parser.set_defaults(
        max_epochs=800,
        accelerator='gpu', strategy='ddp', devices=1, precision=16,
        check_val_every_n_epoch=1,
        # Should be true, set to false since gather() doesn't yet work with DDP
        deterministic=False,
        default_root_dir=os.path.join(OUTPUTS_DIR, 'logs'),
    )
    parser.add_argument('--device_list', nargs='+', type=int, default=None)

    # Logging
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction)

    temp_args, _ = parser.parse_known_args()
    parser = add_method_args(parser, temp_args)
    parser = add_datamodule_args(parser, temp_args)
    parser = add_devices_args(parser, temp_args)

    return parser


def add_method_args(parser: ArgumentParser, temp_args) -> ArgumentParser:
    method_class: Type[SSLMethod] = ssl_methods.registry.get(temp_args.method)
    parser = method_class.add_argparse_args(parser)
    return parser


def add_datamodule_args(parser: ArgumentParser, temp_args) -> ArgumentParser:
    datamodule_class: Type[SSLDataModule] = datamodules.registry.get(temp_args.dataset)
    parser = datamodule_class.add_argparse_args(parser)
    return parser


def add_devices_args(parser: ArgumentParser, temp_args) -> ArgumentParser:
    # TODO: make this nicer
    if temp_args.device_list is not None:
        parser.set_defaults(devices=temp_args.device_list)
        num_devices = len(temp_args.device_list)
    elif temp_args.devices is not None:
        num_devices = int(temp_args.devices)
    else:
        num_devices = 1
    parser.set_defaults(sync_batchnorm=num_devices > 1)
    return parser

def get_method(args) -> SSLMethod:
    method_constructor: Type[SSLMethod] = ssl_methods.registry.get(args.method)
    method_lit_module = method_constructor(**vars(args))
    return method_lit_module


def get_datamodule(args) -> SSLDataModule:
    datamodule_constructor: Type[SSLDataModule] = datamodules.registry.get(args.dataset)
    datamodule = datamodule_constructor(**vars(args))
    return datamodule


def attach_transforms(datamodule: SSLDataModule, method: SSLMethod) -> None:
    datamodule.pretrain_transform = method.pretrain_transform(normalization=datamodule.normalization_transform)


def get_loggers(method, args):
    experiment_name = f"{method}-{args.dataset}"

    wandb_logger = WandbLogger(
        project='joint-embedding',
        name=experiment_name,
        save_dir=OUTPUTS_DIR,
        offline=args.offline,
        log_model=False,
    )
    csv_logger = CSVLogger(
        name='pretrain',
        save_dir=os.path.join(OUTPUTS_DIR, 'csv')
    )

    return [wandb_logger, csv_logger]


def get_callbacks(method: SSLMethod, args):
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    embed_dim = method.embedding_dim()

    knn_online_evaluator = KNNOnlineEvaluator()
    linear_probe_online_evaluator = LinearProbeOnlineEvaluator(
        embed_dim=embed_dim,
        num_classes=args.dataset_num_classes
    )
    online_evaluation_runner = OnlineEvaluationRunner(
        embed_dim=embed_dim,
        online_evaluators=[knn_online_evaluator, linear_probe_online_evaluator]
    )

    return [lr_monitor, online_evaluation_runner]


if __name__ == '__main__':
    main()
