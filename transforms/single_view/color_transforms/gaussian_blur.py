import random

from PIL import ImageFilter


class RandomGaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(.1, 2.)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
