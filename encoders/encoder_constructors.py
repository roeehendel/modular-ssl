from enum import Enum

from encoders.masked_vision_transformer import vit_tiny_masked
from encoders.resnet import resnet_18, resnet_50
from encoders.vision_transformer import vit_tiny, vit_micro


class EncoderType(Enum):
    RESNET = 'resnet'
    VIT = 'vit'
    MASKVIT = 'maskvit'


def get_encoder_type(encoder_model) -> EncoderType:
    encoder_type = encoder_model.split('_')[0]
    encoder_type = EncoderType(encoder_type)
    return encoder_type


_ENCODER_CONSTRUCTORS = {
    EncoderType.RESNET: {
        '18': resnet_18,
        '50': resnet_50,
    },
    EncoderType.VIT: {
        'tiny': vit_tiny,
        'micro': vit_micro,
        'tiny_masked': vit_tiny_masked,
    },
}

ENCODER_CONSTRUCTORS = {
    f'{encoder_type.value}_{encoder_name}': encoder_constructor
    for encoder_type, encoder_constructors in _ENCODER_CONSTRUCTORS.items()
    for encoder_name, encoder_constructor in encoder_constructors.items()
}
