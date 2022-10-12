import random
from typing import Union

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class RandomSubsetDataset(Dataset):
    def __init__(self, original_dataset: Dataset, subset_size: Union[int, float]):
        self.original_dataset = original_dataset
        self.subset_size = subset_size if isinstance(subset_size, int) else int(len(original_dataset) * subset_size)

    def __getitem__(self, index) -> T_co:
        return self.original_dataset[random.randint(0, len(self.original_dataset) - 1)]

    def __len__(self):
        return self.subset_size
