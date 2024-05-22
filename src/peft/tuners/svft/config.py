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
from typing import List, Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class SVFTConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`SVFTModel`].

    Args:
        r (`int`, *optional*, defaults to `None`):
            Rank of the decomposition. If unspecified, the rank will be set to min(input_dim, output_dim).
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply SVFT to. Only linear layers are supported.
        svft_dropout (`float`):
            The dropout probability for SVFT layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for SVFT. Can be 'none', 'all' or 'svft_only'. If 'all' or 'svft_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from SVFT layers to be set as trainable and saved in the final checkpoint.
        init_weights (`Literal["svd", "ortho"]`):
            Method to initialize the weights of the SVFT layers. Can be 'svd' or 'ortho'.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the SVFT transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the SVFT
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: Optional[int] = field(default=None, metadata={"help": "Rank of the decomposition."})

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with SVFT."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layer and Conv1D are supported."
            )
        },
    )
    svft_dropout: float = field(default=0.0, metadata={"help": "SVFT dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for SVFT. Can be 'none', 'all' or 'svft_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from SVFT layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    init_weights: Literal["svd", "ortho"] = field(
        default="svd",
        metadata={
            "help": ("Method to initialize the weights of the SVFT layers. Can be 'svd' or 'ortho'."),
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer"
                " pattern is not in the common layers pattern."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.SVFT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
