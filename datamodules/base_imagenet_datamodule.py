from abc import ABC

from datamodules.ssl_datamodule import SSLDataModule, ImageDims


class BaseImagenetDataModule(SSLDataModule, ABC):
    img_dims = ImageDims(3, 224, 224)
    num_classes = 1000
    normalization_mean = (0.485, 0.456, 0.406)
    normalization_std = (0.229, 0.224, 0.225)
