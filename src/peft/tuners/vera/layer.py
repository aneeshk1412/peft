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

import warnings
from typing import Any, List, Optional

import torch
from torch import nn

from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose


class VeraLayer(LoraLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = (
        "lora_vera_IS",
        "lora_vera_OS",
    )
    # other_param_names is defined in LoraLayer

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.train_IS = {}
        self.train_OS = {}
        self.lora_vera_IS = nn.ParameterDict({})
        self.lora_vera_OS = nn.ParameterDict({})
        self.lora_vera_B = nn.ModuleDict({})
        self.lora_vera_A = nn.ModuleDict({})

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        train_IS=True,
        train_OS=True,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer, but the value passed is {r}")
        ## truncate r
        if r > min(self.in_features, self.out_features):
            r = min(self.in_features, self.out_features)

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        # Actual trainable parameters
        self.train_IS[adapter_name] = train_IS
        self.train_OS[adapter_name] = train_OS

        self.lora_vera_IS[adapter_name] = nn.Parameter(torch.ones(1, r), requires_grad=train_IS)
        self.lora_vera_OS[adapter_name] = nn.Parameter(torch.ones(1, self.out_features), requires_grad=train_OS)

        self.lora_vera_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        self.lora_vera_B[adapter_name].requires_grad_(False)
        self.lora_vera_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_vera_A[adapter_name].requires_grad_(False)

        self.scaling[adapter_name] = 1.0

        self.reset_lora_parameters(adapter_name)

        if hasattr(self.get_base_layer(), "qweight"):
            # QuantLinear
            self.to(self.get_base_layer().qweight.device)
        else:
            self.to(self.get_base_layer().weight.device)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_vera_IS.keys():
            nn.init.kaiming_uniform_(self.lora_vera_B[adapter_name].weight)
            nn.init.kaiming_uniform_(self.lora_vera_A[adapter_name].weight)

            if self.train_IS[adapter_name]:
                nn.init.zeros_(self.lora_vera_IS[adapter_name])
            if self.train_OS[adapter_name]:
                nn.init.zeros_(self.lora_vera_OS[adapter_name])


class VeraLinear(nn.Module, VeraLayer):
    # Vera-based adaptation by a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 1,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        train_IS: bool = True,
        train_OS: bool = True,
        fan_in_fan_out: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        VeraLayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            train_IS=train_IS,
            train_OS=train_OS,
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if active_adapter in self.lora_vera_IS.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_vera_IS.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        ## B (in, r) , IS (r, 1), A (r, out), OS (1, out)
        lora_vera_IS = self.lora_vera_IS[adapter]
        lora_vera_OS = self.lora_vera_OS[adapter]
        lora_vera_A = self.lora_vera_B[adapter].weight
        lora_vera_B = self.lora_vera_A[adapter].weight
        scaling = self.scaling[adapter]
        w = (lora_vera_A @ torch.diag(lora_vera_IS.squeeze()) @ lora_vera_B) @ torch.diag(lora_vera_OS.squeeze())
        return transpose(w.T, self.fan_in_fan_out) * scaling

    def forward_adapter(self, x: torch.Tensor, adapter_name: str) -> torch.Tensor:
        lora_vera_IS = self.lora_vera_IS[adapter_name]
        lora_vera_OS = self.lora_vera_OS[adapter_name]
        lora_vera_A = self.lora_vera_A[adapter_name]
        lora_vera_B = self.lora_vera_B[adapter_name]
        lora_dropout = self.lora_dropout[adapter_name]
        scaling = self.scaling[adapter_name]
        return lora_vera_OS * lora_vera_B(lora_vera_IS * lora_vera_A(lora_dropout(x))) * scaling

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_vera_IS.keys():
                    continue

                x = x.to(self.lora_vera_IS[active_adapter].dtype)
                result += self.forward_adapter(x, active_adapter)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "vera." + rep
