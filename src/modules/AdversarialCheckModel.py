import torch
from torch import Tensor, nn


class AdversarialCheckModel(nn.Module):
    def __init__(self, model: nn.Module, ground_truth_neuron_index: int) -> None:
        """
        Args:
            model (nn.Module): The Pytorch neural network.
            ground_truth_neuron_index (int): Index of the model's output neuron \
                that's chosen to be the "ground-truth".
        """
        super().__init__()
        self.model = model
        self.ground_truth_neuron_index = ground_truth_neuron_index

    def forward(self, batched_concrete_inputs: Tensor) -> bool:
        """Returns whether any of the concrete inputs falsifies the problem.

        Specifically, returns `True` if any `y_i - y_g >= 0`, where:
        - `y_g` is the "ground-truth" neuron's output
        - `y_i` is any other output-neuron.
        """
        pred = self.model.forward(batched_concrete_inputs)
        assert isinstance(pred, Tensor)

        num_batches, num_output = pred.shape
        i = self.ground_truth_neuron_index
        ground_truth = pred[:, i : i + 1]
        rest_of_pred = torch.cat((pred[:, :i], pred[:, i + 1 :]), dim=1)
        assert ground_truth.shape == (num_batches, 1)
        assert rest_of_pred.shape == (num_batches, num_output - 1)

        return bool(torch.any(rest_of_pred - ground_truth >= 0).item())
