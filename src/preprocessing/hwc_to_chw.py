from typing import Tuple, Union

import torch
from torch import Tensor
from typing_extensions import TypeAlias

CNNShape: TypeAlias = Union[Tuple[int, int, int], torch.Size]


def flatten_hwc_to_chw(X: Tensor, hwc_shape: CNNShape, permute_on_dim=0) -> Tensor:
    assert X.dim() > permute_on_dim
    assert X.dim() <= 2

    if X.dim() == 1:
        return X.reshape(hwc_shape).permute(2, 0, 1).flatten()

    if permute_on_dim == 0:
        output = X.reshape(hwc_shape + (-1,)).permute(2, 0, 1, 3).flatten(start_dim=0, end_dim=2)
        assert output.dim() == X.dim()
        return output
    elif permute_on_dim == 1:
        output = X.reshape((-1,) + hwc_shape).permute(0, 3, 1, 2).flatten(start_dim=1, end_dim=3)
        assert output.dim() == X.dim()
        return output
    raise NotImplementedError()


def flatten_unstable_hwc_to_chw(
    unstable_only: Tensor,
    unstable_mask: Tensor,
    hwc_shape: CNNShape,
    mask_dim: int = 0,
) -> Tensor:
    full_tensor_shape = list(unstable_only.shape)
    full_tensor_shape[mask_dim] = unstable_mask.size(0)

    # Setting the mask on the correct dimension.
    unstable_mask = unstable_mask[(None,) * mask_dim + (...,)]

    dtype = unstable_only.dtype
    full_tensor = torch.full(full_tensor_shape, torch.finfo(dtype).max, dtype=dtype)
    full_tensor[unstable_mask.expand(full_tensor_shape)] = unstable_only.flatten()
    permuted_full = flatten_hwc_to_chw(full_tensor, hwc_shape, permute_on_dim=mask_dim)
    permuted_mask = flatten_hwc_to_chw(unstable_mask, hwc_shape, permute_on_dim=mask_dim)
    return permuted_full[permuted_mask.expand(full_tensor_shape)].reshape(unstable_only.shape)
