from functools import reduce
from typing import Tuple, Union

import torch
from torch import Tensor
from typing_extensions import TypeAlias

CNNShape: TypeAlias = Union[Tuple[int, int, int], torch.Size]


def flattened_hwc_to_chw(X: Tensor, hwc_shape: CNNShape, permute_on_dim=0) -> Tensor:
    """Takes in a flattened CNN input (or a batch of them) in the
    Height-Width-Channel (HWC) format, then unflattens it back to HWC, converts
    it to Channel-Height-Width (CHW) format, and then re-flattens it.

    Currently only handles 1D or 2D tensors.

    Args:
        X (Tensor): 1D tensor in HWC format, or 2D where one of the dim is in HWC format.
        hwc_shape (CNNShape): The supposed unflattened shape of the HWC-formatted dim.
        permute_on_dim (int, optional): For 2D tensors; used to indicate which \
            dim is the one in HWC format. Defaults to 0.

    Returns:
        Tensor: Tensor with same shape as `X` but `dim=permute_on_dim` is in \
            flattened-CHW format, instead of flattened-HWC.
    """
    assert X.dim() > permute_on_dim
    assert X.dim() <= 2
    assert reduce(lambda x, y: x * y, hwc_shape) == X.size(permute_on_dim)

    # For unbatched.
    if X.dim() == 1:
        return X.reshape(hwc_shape).permute(2, 0, 1).flatten()

    # Convert HWC -> CHW on dim 0, ignoring dim 1.
    if permute_on_dim == 0:
        output = X.reshape(hwc_shape + (-1,)).permute(2, 0, 1, 3).flatten(start_dim=0, end_dim=2)
        assert output.dim() == X.dim()
        return output

    # Convert HWC -> CHW on dim 1, ignoring dim 0.
    elif permute_on_dim == 1:
        output = X.reshape((-1,) + hwc_shape).permute(0, 3, 1, 2).flatten(start_dim=1, end_dim=3)
        assert output.dim() == X.dim()
        return output

    raise NotImplementedError()


def flattened_unstable_hwc_to_chw(
    unstable_only: Tensor,
    hwc_unstable_mask: Tensor,
    hwc_shape: CNNShape,
    mask_dim: int = 0,
) -> Tensor:
    """Takes in a flattened Height-Width-Channel (HWC) formatted tensor (or a
    batch of them) that has been masked to only include the unstable neurons;
    then unmasks it to include all neurons, unflattens it back to HWC,
    converts it to Channel-Height-Width (CHW) format, re-flattens it, then
    re-mask it to only include unstable neurons.

    Currently only handles `unstable_only` being in 1D or 2D.

    Args:
        unstable_only (Tensor): Unstable-neuron-masked 1D tensor in HWC format, \
            or 2D where one of the dim is in masked HWC format.
        hwc_unstable_mask (Tensor): Mask in flattened-HWC format, selecting \
            only the unstable neurons.
        hwc_shape (CNNShape): The supposed unflattened shape of the unmasked HWC-formatted dim.
        mask_dim (int, optional): For 2D tensors; used to indicate which \
            dim is the one in masked HWC format. Defaults to 0.

    Returns:
        Tensor: Tensor with same shape as `unstable_only` but `dim=mask_dim` is \
            in masked-flattened-CHW format, instead of masked-flattened-HWC.
    """
    assert hwc_unstable_mask.dim() == 1
    assert reduce(lambda x, y: x * y, hwc_shape) == len(hwc_unstable_mask)

    full_tensor_shape = list(unstable_only.shape)
    full_tensor_shape[mask_dim] = len(hwc_unstable_mask)

    # Setting the mask on the correct dimension.
    hwc_unstable_mask = hwc_unstable_mask[(None,) * mask_dim + (...,)]

    dtype = unstable_only.dtype
    full_tensor = torch.full(full_tensor_shape, torch.finfo(dtype).max, dtype=dtype)
    full_tensor[hwc_unstable_mask.expand(full_tensor_shape)] = unstable_only.flatten()
    permuted_full = flattened_hwc_to_chw(full_tensor, hwc_shape, permute_on_dim=mask_dim)
    permuted_mask = flattened_hwc_to_chw(hwc_unstable_mask, hwc_shape, permute_on_dim=mask_dim)
    return permuted_full[permuted_mask.expand(full_tensor_shape)].reshape(unstable_only.shape)
