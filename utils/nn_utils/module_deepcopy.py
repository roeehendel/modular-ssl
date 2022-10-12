import copy

from torch import nn
from torch.nn.utils.weight_norm import WeightNorm


def module_deepcopy(original_module: nn.Module):
    for module in original_module.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                delattr(module, hook.name)
    copy_module = copy.deepcopy(original_module)
    for module in original_module.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                hook(module, None)
    return copy_module
