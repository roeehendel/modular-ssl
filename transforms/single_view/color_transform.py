import argparse

from torchvision.transforms import transforms, RandomApply

from transforms.single_view.color_transforms.gaussian_blur import RandomGaussianBlur
from transforms.single_view.color_transforms.standard_color_jitter import StandardColorJitter


class ColorTransform(object):
    def __init__(self,
                 jitter_p: float = 0.8, jitter_strength: float = 0.5,
                 grayscale_p: float = 0.2,
                 gaussian_blur_p: float = 0.0,
                 solarize_p: float = 0.0, **kwargs) -> None:
        color_jitter = RandomApply([StandardColorJitter(jitter_strength=jitter_strength)], p=jitter_p)
        grayscale = transforms.RandomGrayscale(p=grayscale_p)
        gaussian_blur = RandomApply([RandomGaussianBlur()], p=gaussian_blur_p)
        solarize = RandomApply([transforms.RandomSolarize(threshold=0.5)], p=solarize_p)

        self.transform = transforms.Compose([
            color_jitter,
            grayscale,
            gaussian_blur,
            solarize
        ])

    @classmethod
    def add_argparse_args(cls, parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)

        parser.add_argument("--jitter_p", type=float, default=0.8)
        parser.add_argument("--jitter_strength", type=float, default=0.5)
        parser.add_argument("--grayscale_p", type=float, default=0.2)
        parser.add_argument("--gaussian_blur_p", type=float, default=0.0)
        parser.add_argument("--solarize_p", type=float, default=0.0)

        return parent_parser

    def __call__(self, sample):
        return self.transform(sample)
