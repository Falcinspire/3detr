# Encoder that merely forwards the state it's given. This is to be used
# as a lower-bound for the performance of the FNet integration.

from typing import Optional

import torch.utils.checkpoint
from torch import Tensor, nn
from torch import nn

class NoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    #TODO can these other arguments really just be ignored?
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                xyz: Optional [Tensor] = None,
                transpose_swap: Optional[bool] = False,
                ):
        return xyz, src, None