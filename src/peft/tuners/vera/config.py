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

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType


@dataclass
class VeraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Vera`].

    Args:
        train_OS (`bool`):
            Set this to True if the outer scaling is to be trained. Defaults to True.
        train_IS (`bool`):
            Set this to True if the inner scaling is to be trained. Defaults to True.
    """

    train_OS: bool = field(
        default=True,
        metadata={"help": "Set this to True if the outer scaling is to be trained. Defaults to True."},
    )
    train_IS: bool = field(
        default=True,
        metadata={"help": "Set this to True if the inner scaling is to be trained. Defaults to True."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.VERA
        self.init_lora_weights = False
