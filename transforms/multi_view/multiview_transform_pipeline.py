from abc import ABC, abstractmethod


class MultiviewTransform(ABC):
    @abstractmethod
    def __call__(self, samples: list) -> list:
        pass


class IIDMultiviewTransform(MultiviewTransform):
    def __init__(self, transform: callable = None) -> None:
        self.transform = transform

    def __call__(self, samples: list) -> list:
        return [self.transform(sample) for sample in samples]


class MultiviewTransformPipeline:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: list) -> list:
        samples = [sample]
        for transform in self.transforms:
            if not isinstance(transform, MultiviewTransform):
                transform = IIDMultiviewTransform(transform)
            samples = transform(samples)
        return samples
