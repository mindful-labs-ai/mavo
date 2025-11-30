import functools
import torch


def patch_torch_load():
    """
    Patch torch.load to use weights_only=False by default.
    """
    _original_torch_load = torch.load

    @functools.wraps(_original_torch_load)
    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    torch.serialization.load = _patched_torch_load


patch_torch_load()
