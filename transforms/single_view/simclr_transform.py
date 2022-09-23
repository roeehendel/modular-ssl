from torchvision.transforms import transforms


class SimCLRTransform:
    """
    Standard image augmentation-based transform
    Adapted from lightning-bolts implementation of SimCLR single_view
    """

    def __init__(
            self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1.0, normalize=None,
            crop_scale=(0.2, 1.0),
    ) -> None:
        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height, scale=crop_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5))

        self.data_transforms = transforms.Compose(data_transforms)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.transform = transforms.Compose([self.data_transforms, self.final_transform])

    def __call__(self, sample):
        return self.transform(sample)
