from typing import Optional

from torchvision.transforms import transforms, InterpolationMode


class EvaluationTransform:
    def __init__(self, input_height: int = 32, normalization: Optional = None):
        self.input_height = input_height

        if normalization is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalization])

        self.transform = transforms.Compose(
            [
                transforms.Resize(int(self.input_height * 1.1), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.input_height),
                self.final_transform,
            ]
        )

    def __call__(self, sample):
        return self.transform(sample)
