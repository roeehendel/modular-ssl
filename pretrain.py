import argparse
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from config import OUTPUTS_DIR
from data.data_modules.imagenet_datamodule import ImagenetDataModule
from encoders.encoder_constructors import get_encoder_type, ENCODER_CONSTRUCTORS, EncoderType
from joint_embedding_methods.dino import DINO
from joint_embedding_methods.joint_embedding_method import JointEmbeddingMethod
from joint_embedding_methods.mask_dino import MaskDINO
from joint_embedding_methods.simsiam import SimSiam
from online_evaluators.knn_online_evaluator import KNNOnlineEvaluator
from online_evaluators.linear_online_evaluator import LinearProbeOnlineEvaluator
from online_evaluators.online_evaluation_runner import OnlineEvaluationRunner
from transforms.single_view.evaluation_transform import EvaluationTransform
from transforms.train_and_eval_transform import TrainAndEvalTransform


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    print(args.__dict__)

    seed_everything(args.random_seed, workers=True)

    add_dataset_args(args)

    encoder = get_encoder(args)
    method_module = get_method(args, encoder)
    datamodule = get_data(args, method_module)
    loggers = get_loggers(args)
    callbacks = get_callbacks(args)

    if args.device_list is not None:
        devices = args.device_list
        using_multiple_gpus = len(devices) > 1
    else:
        devices = args.devices
        using_multiple_gpus = args.devices is not None and int(args.devices) > 1

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=loggers,
        callbacks=callbacks,
        # TODO: move the following to args
        default_root_dir=os.path.join(OUTPUTS_DIR, 'logs'),
        devices=devices,
        deterministic=False,  # Should be true, set to false since gather() doesn't yet work with DDP
        sync_batchnorm=using_multiple_gpus,
    )
    trainer.fit(model=method_module, datamodule=datamodule)


def get_arg_parser():
    parser = ArgumentParser()

    # Random seed
    parser.add_argument('--random_seed', type=int, default=42)

    # Data
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Pretraining dataset", choices=["cifar10", "imagenet"])
    # TODO: change this to the regular imagenet path
    parser.add_argument("--imagenet_dir", type=str, default="/home/gamir/datasets/ilsvrc",
                        help="Path to ImageNet (ILSVRC2012) dataset")
    parser.add_argument("--data_dir", type=str, default="/home/gamir/hendel/datasets/",
                        help="Path to download data")

    # Transforms
    # parser.add_argument("--gaussian_blur", action="store_false", help="add gaussian blur")
    # parser.add_argument("--jitter_strength", type=float, default=0.5, help="jitter strength")

    # Model
    parser.add_argument("--encoder_model", type=str, default="resnet_18", help="The encoder model to use",
                        choices=ENCODER_CONSTRUCTORS.keys())

    # Joint Embedding Method
    parser = JointEmbeddingMethod.add_model_specific_args(parser)
    parser.add_argument("--method", type=str, default="simsiam", help="The joint embedding method to use",
                        choices=["simsiam", "dino", "mask_dino"])

    # Training
    parser = Trainer.add_argparse_args(parser)
    # TODO: check if it's better to use strategy='ddp_find_unused_parameters_false'
    parser.set_defaults(
        max_epochs=800,
        accelerator='gpu', strategy='ddp', devices=1,
        precision=16,
        check_val_every_n_epoch=1
    )
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")
    parser.add_argument('--device_list', nargs='+', type=int, default=None)

    # Logging
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction)

    temp_args, _ = parser.parse_known_args()

    # Model-type-specific args
    encoder_type = get_encoder_type(temp_args.encoder_model)
    if encoder_type == EncoderType.RESNET:
        default_first_conv = {'cifar10': False, 'imagenet': True}[temp_args.dataset]
        default_maxpool1 = {'cifar10': False, 'imagenet': True}[temp_args.dataset]
        parser.add_argument("--first_conv", action=argparse.BooleanOptionalAction)
        parser.add_argument("--maxpool1", action=argparse.BooleanOptionalAction)
        parser.set_defaults(first_conv=default_first_conv, maxpool1=default_maxpool1)
    elif encoder_type == EncoderType.VIT:
        default_patch_size = {'cifar10': 4, 'imagenet': 16, }[temp_args.dataset]
        parser.add_argument('--patch_size', type=int, default=default_patch_size)
        parser.add_argument("--freeze_patch_embeddings", action=argparse.BooleanOptionalAction,
                            help="Freeze the patch embeddings of the ViT encoder (as per MoCo-v3)")
        parser.set_defaults(freeze_patch_embeddings=False)

    # Method-specific args
    if temp_args.method == 'simsiam':
        parser = SimSiam.add_model_specific_args(parser)
    elif temp_args.method == 'dino':
        parser = DINO.add_model_specific_args(parser)
    elif temp_args.method == 'mask_dino':
        parser = MaskDINO.add_model_specific_args(parser)

    return parser


def add_dataset_args(args):
    if args.dataset == 'cifar10':
        args.input_height = 32
        args.num_classes = 10
        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == 'imagenet':
        args.input_height = 224
        args.num_classes = 1000
        args.gaussian_blur = True
        args.jitter_strength = 1.0
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


def get_encoder(args):
    if args.encoder_model not in ENCODER_CONSTRUCTORS:
        raise ValueError(f'Unknown encoder: {args.encoder_model}')

    encoder_type = get_encoder_type(args.encoder_model)
    encoder_constructor = ENCODER_CONSTRUCTORS[args.encoder_model]

    if encoder_type == EncoderType.RESNET:
        encoder = encoder_constructor(first_conv=args.first_conv, maxpool1=args.maxpool1)
        args.embedding_dim = encoder[0].fc.in_features
    elif encoder_type == EncoderType.VIT:
        encoder = encoder_constructor(img_size=args.input_height, patch_size=args.patch_size)
        args.embedding_dim = encoder.embed_dim

        if args.freeze_patch_embeddings:
            encoder.patch_embed.requires_grad = False
    else:
        raise ValueError(f'Unknown encoder type: {encoder_type}')

    return encoder


def get_method(args, encoder) -> JointEmbeddingMethod:
    method_constructors = {
        'simsiam': SimSiam,
        'dino': DINO,
        'mask_dino': MaskDINO
    }

    if args.method not in method_constructors:
        raise ValueError(f'Unknown joint embedding method: {args.method}')

    method_lit_module = method_constructors[args.method](
        encoder,
        **args.__dict__,
    )

    return method_lit_module


def get_data(args, method_module: JointEmbeddingMethod):
    if args.dataset == 'cifar10':
        val_split = 5000
        datamodule = CIFAR10DataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=val_split,
            drop_last=True
        )
    elif args.dataset == 'imagenet':
        datamodule = ImagenetDataModule(
            data_dir=args.imagenet_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
        )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    normalization = {
        'cifar10': cifar10_normalization(),
        'imagenet': imagenet_normalization()
    }[args.dataset]
    branches_views_transform = method_module.branches_views_transform(
        input_height=args.input_height,
        normalization=normalization
    )
    eval_transform = EvaluationTransform(
        input_height=args.input_height,
        normalization=normalization,
    )
    datamodule.train_transforms = TrainAndEvalTransform(branches_views_transform, eval_transform)
    datamodule.val_transforms = eval_transform

    return datamodule


def get_loggers(args):
    wandb_logger = WandbLogger(
        project='joint-embedding',
        name=f'pretrain-{args.method}-{args.encoder_model}-{args.dataset}',
        save_dir=OUTPUTS_DIR,
        offline=args.offline,
    )
    csv_logger = CSVLogger(
        name='pretrain',
        save_dir=os.path.join(OUTPUTS_DIR, 'csv')
    )

    return [wandb_logger, csv_logger]


def get_callbacks(args):
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    knn_online_evaluator = KNNOnlineEvaluator()
    linear_probe_online_evaluator = LinearProbeOnlineEvaluator(
        embedding_dim=args.embedding_dim,
        num_classes=args.num_classes
    )
    online_evaluation_runner = OnlineEvaluationRunner(
        embedding_dim=args.embedding_dim,
        online_evaluators=[knn_online_evaluator, linear_probe_online_evaluator]
    )

    return [lr_monitor, online_evaluation_runner]


if __name__ == '__main__':
    main()
