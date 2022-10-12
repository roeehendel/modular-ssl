from typing import Tuple, List, Dict

from torchvision import datasets

from utils.file_cache import file_cache


class CachedImageFolder(datasets.ImageFolder):
    @file_cache('classes.json')
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return super().find_classes(directory)

    @file_cache('samples.json')
    def make_dataset(self, directory, *args, **kwargs):
        return super().make_dataset(directory, *args, **kwargs)
