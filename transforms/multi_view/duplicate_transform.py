import itertools

from transforms.multi_view.multiview_transform_pipeline import MultiviewTransform


class DuplicateTransform(MultiviewTransform):
    def __init__(self, num_copies: int = 2) -> None:
        self.num_copies = num_copies

    def __call__(self, samples: list) -> list:
        return list(itertools.chain(*[samples for _ in range(self.num_copies)]))
