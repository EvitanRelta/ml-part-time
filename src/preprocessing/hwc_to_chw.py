from functools import reduce
from typing import Tuple, Union

import torch
from torch import Tensor
from typing_extensions import TypeAlias

CNNShape: TypeAlias = Union[Tuple[int, int, int], torch.Size]


def flatten_hwc_to_chw(X: Tensor, hwc_shape: CNNShape, permute_on_dim=0) -> Tensor:
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


def flatten_unstable_hwc_to_chw(
    unstable_only: Tensor,
    hwc_unstable_mask: Tensor,
    hwc_shape: CNNShape,
    mask_dim: int = 0,
) -> Tensor:
    assert hwc_unstable_mask.dim() == 1
    assert reduce(lambda x, y: x * y, hwc_shape) == len(hwc_unstable_mask)

    full_tensor_shape = list(unstable_only.shape)
    full_tensor_shape[mask_dim] = len(hwc_unstable_mask)

    # Setting the mask on the correct dimension.
    hwc_unstable_mask = hwc_unstable_mask[(None,) * mask_dim + (...,)]

    dtype = unstable_only.dtype
    full_tensor = torch.full(full_tensor_shape, torch.finfo(dtype).max, dtype=dtype)
    full_tensor[hwc_unstable_mask.expand(full_tensor_shape)] = unstable_only.flatten()
    permuted_full = flatten_hwc_to_chw(full_tensor, hwc_shape, permute_on_dim=mask_dim)
    permuted_mask = flatten_hwc_to_chw(hwc_unstable_mask, hwc_shape, permute_on_dim=mask_dim)
    return permuted_full[permuted_mask.expand(full_tensor_shape)].reshape(unstable_only.shape)
