from torch import nn


class MaskDINOTeacher(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x, target_idx):
        x = self.encoder.forward_features(x, target_idx)
        x = self.head(x)
        return x
