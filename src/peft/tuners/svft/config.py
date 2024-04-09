# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Literal

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType


@dataclass
class SVFTConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.SVFT`].

    Args:
        train_A (`bool`):
            Set this to True if the left singular vectors of the weights are to be trained. Defaults to False.
        train_B (`bool`):
            Set this to True if the right singular vectors of the weights are to be trained. Defaults to False.
        init_svft_weights (`Literal["svd", "s_kunif", "uv_kunif", "suv_kunif"]`):
            How to initialize the weights of the SVFT layer. Defaults to "svd".
            svd: Initialize the left and right singular vectors using SVD of the weight and the singular values to zero.
            s_kunif: Initialize the singular values using a kaiming uniform distribution and the left and right singular vectors using the SVD of the weight matrix.
            uv_kunif: Initialize the left and right singular vectors using a kaiming uniform distribution and the singular values to zero.
            suv_kunif: Initialize the left and right singular vectors and singular values using a kaiming uniform distribution.
    """
    train_A: bool = field(
        default=False,
        metadata={"help": "Set this to True if the left singular vectors of the weights are to be trained. Defaults to False."}
    )
    train_B: bool = field(
        default=False,
        metadata={"help": "Set this to True if the right singular vectors of the weights are to be trained. Defaults to False."}
    )
    init_weights: Literal["svd", "s_kunif", "uv_kunif", "suv_kunif"] = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the SVFT layer. Defaults to 'svd'. "
                "svd: Initialize the left and right singular vectors using SVD of the weight and the singular values to zero. "
                "s_kunif: Initialize the singular values using a kaiming uniform distribution and the left and right singular vectors using the SVD of the weight matrix. "
                "suv_kunif: Initialize the left and right singular vectors and singular values using a kaiming uniform distribution. "
            ),
        },
    )


    def __post_init__(self):
        self.peft_type = PeftType.SVFT
        self.init_lora_weights = False
