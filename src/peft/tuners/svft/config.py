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
        r (`int` | None):
            The rank of the SVD approximation. Defaults to None which does a full-rank approximation.
        train_delta_S (`bool`):
            Set this to True if the singular values of the weights are to be trained. Defaults to True.
        init_U (`Literal["svd", "kunif"]`):
            How to initialize the left singular vectors of the weights. Defaults to "svd".
        init_V (`Literal["svd", "kunif"]`):
            How to initialize the right singular vectors of the weights. Defaults to "svd".
        init_delta_S (`Literal["zero", "kunif"]`):
            How to initialize the singular values of the weights. Defaults to "zero".
        gate_delta_S (`bool`):
            Set this to True if you want to use the gating mechanism on the singular values. Defaults to False.
        rank_r (`int` | `None`):
            Adds additional rank r LoRA layers to the model. Defaults to None.
        gate_rank_r (`bool`):
            Set this to True if you want to use the gating mechanism on the rank r addition. Defaults to False.
    """

    r: int = field(
        default=None,
        metadata={"help": "The rank of the SVD approximation. Defaults to None which does a full-rank approximation."},
    )

    train_delta_S: bool = field(
        default=True,
        metadata={
            "help": "Set this to True if the singular values of the weights are to be trained. Defaults to True."
        },
    )

    init_U: Literal["svd", "kunif"] = field(
        default="svd",
        metadata={"help": "How to initialize the left singular vectors of the weights. Defaults to 'svd'."},
    )

    init_V: Literal["svd", "kunif"] = field(
        default="svd",
        metadata={"help": "How to initialize the right singular vectors of the weights. Defaults to 'svd'."},
    )

    init_delta_S: Literal["zero", "kunif"] = field(
        default="zero",
        metadata={"help": "How to initialize the singular values of the weights. Defaults to 'zero'."},
    )

    gate_delta_S: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if you want to use the gating mechanism on the singular values. Defaults to False."
        },
    )

    rank_r: int = field(
        default=None,
        metadata={"help": "Adds additional rank r LoRA layers to the model. Defaults to None."},
    )

    gate_rank_r: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if you want to use the gating mechanism on the rank r addition. Defaults to False."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.SVFT
        self.init_lora_weights = False
