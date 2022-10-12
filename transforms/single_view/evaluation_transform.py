from typing import Optional

from torchvision.transforms import transforms, InterpolationMode


class EvaluationTransform:
    def __init__(self, input_height: int = 32, resize_factor: float = 1.0, normalization: Optional = None, **kwargs):
        self.input_height = input_height

        transform_list = []

        if resize_factor != 1.0:
            resize = transforms.Resize(size=int(input_height * resize_factor), interpolation=InterpolationMode.BICUBIC)
            transform_list.append(resize)

        transform_list += [
            transforms.CenterCrop(self.input_height),
            transforms.ToTensor(),
        ]

        if normalization is not None:
            transform_list.append(normalization)

        self.transform = transforms.Compose(transform_list)

    def __call__(self, sample):
        return self.transform(sample)
