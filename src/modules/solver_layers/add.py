from torch import nn
from typing_extensions import override


class Add_SL(nn.Identity):
    @override
    def __init__(self) -> None:
        super().__init__()
