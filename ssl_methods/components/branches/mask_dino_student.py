from torch import nn


class MaskDINOStudent(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, head: nn.Module, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head

    def forward(self, x, input_idx, target_idx):
        x = self.encoder.forward_features(x, input_idx)
        x = self.decoder(x, input_idx, target_idx)  # TODO: uncomment this line
        x = self.head(x)
        return x
