import argparse
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised.resnets import resnet18
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch import nn

from config import OUTPUTS_DIR
from encoders.masked_vision_transformer import masked_vit_tiny
from joint_embedding_methods.dino import DINO
from joint_embedding_methods.joint_embedding_method import JointEmbeddingMethod
from joint_embedding_methods.simsiam import SimSiam
from transforms.single_view.evaluation_transform import EvaluationTransform
from transforms.train_and_eval_transform import TrainAndEvalTransform
from utils.lambda_module import LambdaModule


def get_arg_parser():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # Data
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Pretraining dataset", choices=["cifar10", "imagenet"])
    parser.add_argument("--imagenet_path", type=str, default="/home/gamir/datasets/ilsvrc100",
                        help="Path to ImageNet (ILSVRC2012) dataset")
    parser.add_argument("--data_dir", type=str, default="/home/gamir/hendel/datasets/",
                        help="path to download data")

    # Transforms
    # parser.add_argument("--gaussian_blur", action="store_false", help="add gaussian blur")
    # parser.add_argument("--jitter_strength", type=float, default=0.5, help="jitter strength")

    # Model
    parser.add_argument("--encoder_type", type=str, default="resnet", help="The encoder model to use",
                        choices=["resnet", "vit", "maskvit"])
    temp_args, _ = parser.parse_known_args()
    if temp_args.encoder_type == 'vit':
        parser.add_argument("--freeze_patch_embeddings", action=argparse.BooleanOptionalAction,
                            help="Freeze the patch embeddings of the ViT encoder (as per MoCo-v3)")
        parser.set_defaults(freeze_patch_embeddings=False)

    # Joint Embedding Method
    parser.add_argument("--method", type=str, default="simsiam", help="The joint embedding method to use")

    # Training
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")
    parser.set_defaults(check_val_every_n_epoch=2)

    # Optimization
    # parser.add_argument("--base_lr", default=5e-4, type=float, help="base learning rate")
    # parser.add_argument("--weight_decay", default=0.04, type=float, help="weight decay")
    # parser.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum")
    # parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")

    # Logging
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction)

    temp_args, _ = parser.parse_known_args()
    if temp_args.method == 'simsiam':
        parser = SimSiam.add_model_specific_args(parser)
    elif temp_args.method == 'dino':
        parser = DINO.add_model_specific_args(parser)

    return parser


def get_data(args, method_module: JointEmbeddingMethod):
    if args.dataset == 'cifar10':
        val_split = 5000
        datamodule = CIFAR10DataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=val_split,
        )

        # dataset_train_knn = datamodule.dataset_train

        normalization = cifar10_normalization()
        args.gaussian_blur = False
        args.jitter_strength = 0.5
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    branches_views_transform = method_module.branches_views_transform(
        input_height=args.input_height,
        normalization=normalization
    )
    eval_transform = EvaluationTransform(
        input_height=args.input_height,
        normalization=normalization,
    )
    # dataset_train_knn.transform = eval_transform
    datamodule.val_transforms = eval_transform
    datamodule.train_transforms = TrainAndEvalTransform(branches_views_transform, eval_transform)

    return datamodule


def get_encoder(args):
    if args.encoder_type == 'resnet':
        encoder = nn.Sequential(
            resnet18(first_conv=False, maxpool1=False, return_all_feature_maps=False),
            LambdaModule(lambda x: x[0]),
        )
        encoder.embedding_dim = encoder[0].fc.in_features
    elif args.encoder_type == 'vit':
        from encoders.vision_transformer import vit_super_tiny
        encoder = vit_super_tiny(img_size=args.input_height, patch_size=4)
        encoder.embedding_dim = encoder.embed_dim

        if args.freeze_patch_embeddings:
            # freeze patch embedding as per MoCo-v3
            encoder.patch_embed.requires_grad = False

    elif args.encoder_type == 'maskvit':
        encoder = nn.Sequential(
            masked_vit_tiny(img_size=args.input_height, patch_size=4),
            LambdaModule(lambda x: x[:, 0, :]),
        )
        encoder.embedding_dim = encoder[0].embed_dim
    else:
        raise ValueError(f'Unknown encoder: {args.dataset}')

    return encoder


def get_method(args, encoder) -> JointEmbeddingMethod:
    if args.method == 'simsiam':
        lit_module = SimSiam(
            encoder,
            encoder_embedding_dim=encoder.embedding_dim,
            **args.__dict__,
            # base_lr=args.base_lr,
            # momentum=args.momentum,
            # weight_decay=args.weight_decay,
            # warmup_epochs=args.warmup_epochs,
        )
    elif args.method == 'dino':
        lit_module = DINO(
            encoder,
            encoder_embedding_dim=encoder.embedding_dim,
            **args.__dict__,
            # base_lr=args.base_lr,
            # weight_decay=args.weight_decay,
            # warmup_epochs=args.warmup_epochs,
        )
    else:
        raise ValueError(f'Unknown joint embedding method: {args.joint_embedding_method}')

    return lit_module


def get_loggers(args):
    wandb_logger = WandbLogger(
        project='joint-embedding',
        name=f'pretrain-{args.method}-{args.encoder_type}-{args.dataset}',
        save_dir=OUTPUTS_DIR,
        offline=args.offline,
    )
    csv_logger = CSVLogger(
        name='pretrain',
        save_dir=os.path.join(OUTPUTS_DIR, 'csv')
    )

    return [wandb_logger, csv_logger]


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    args.input_height = {
        'cifar10': 32,
    }[args.dataset]

    encoder = get_encoder(args)
    method_module = get_method(args, encoder)
    datamodule = get_data(args, method_module)

    loggers = get_loggers(args)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    using_multiple_gpus = args.devices is not None and int(args.devices) > 1
    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=os.path.join(OUTPUTS_DIR, 'logs'),
        sync_batchnorm=using_multiple_gpus,
        logger=loggers,
        callbacks=[lr_monitor],
    )
    trainer.fit(model=method_module, datamodule=datamodule)


if __name__ == '__main__':
    main()
