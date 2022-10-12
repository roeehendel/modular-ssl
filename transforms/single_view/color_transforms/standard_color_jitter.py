from torchvision.transforms import transforms


class StandardColorJitter(object):
    def __init__(self, jitter_strength: float = 0.5):
        self.jitter_strength = jitter_strength

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

    def __call__(self, x):
        return self.color_jitter(x)
