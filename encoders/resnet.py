from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from torch import nn

from utils.lambda_module import LambdaModule


def _wrap_pl_bolts_resnet(resnet_constructor, **kwargs):
    return nn.Sequential(
        resnet_constructor(**kwargs),
        LambdaModule(lambda x: x[0]),
    )


def resnet_18(first_conv: bool = True, maxpool1: bool = True, return_all_feature_maps: bool = False, **kwargs):
    return _wrap_pl_bolts_resnet(resnet18, first_conv=first_conv, maxpool1=maxpool1,
                                 return_all_feature_maps=return_all_feature_maps, **kwargs)


def resnet_50(first_conv: bool = True, maxpool1: bool = True, return_all_feature_maps: bool = False, **kwargs):
    return _wrap_pl_bolts_resnet(resnet50, first_conv=first_conv, maxpool1=maxpool1,
                                 return_all_feature_maps=return_all_feature_maps, **kwargs)
