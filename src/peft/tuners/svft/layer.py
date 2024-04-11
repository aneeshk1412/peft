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
from math import sqrt
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose


class SVFTLayer(LoraLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("lora_S", "lora_rank_one_At", "lora_rank_one_B", "lora_gate_S", "lora_gate_rank_one")
    # other_param_names is defined in LoraLayer

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.lora_Ut = nn.ParameterDict({})
        self.lora_V = nn.ParameterDict({})
        self.lora_S = nn.ParameterDict({})
        self.lora_rank_one_At = nn.ParameterDict({})
        self.lora_rank_one_B = nn.ParameterDict({})
        self.lora_gate_S = nn.ParameterDict({})
        self.lora_gate_rank_one = nn.ParameterDict({})

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_weights,
        train_A=False,
        train_B=False,
        s_gating=False,
        rank_one=False,
        rank_one_gating=False,
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
        self.lora_V[adapter_name] = nn.Parameter(torch.randn(self.in_features, r), requires_grad=train_B)
        self.lora_S[adapter_name] = nn.Parameter(torch.randn(r, 1))
        self.lora_Ut[adapter_name] = nn.Parameter(torch.randn(r, self.out_features), requires_grad=train_A)

        if s_gating:
            self.lora_gate_S[adapter_name] = nn.Parameter(torch.zeros(1), requires_grad=True)

        if rank_one:
            self.lora_rank_one_B[adapter_name] = nn.Parameter(torch.randn(self.in_features, 1), requires_grad=True)
            self.lora_rank_one_At[adapter_name] = nn.Parameter(torch.randn(1, self.out_features), requires_grad=True)

        if rank_one and rank_one_gating:
            self.lora_gate_rank_one[adapter_name] = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Scaling factor (square root of the rank)
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else 1.0 / sqrt(r)

        self.reset_lora_parameters(adapter_name, init_weights)

        # For safety, we freeze the weights again
        self.lora_Ut[adapter_name].requires_grad = train_A
        self.lora_V[adapter_name].requires_grad = train_B

        if hasattr(self.get_base_layer(), "qweight"):
            # QuantLinear
            self.to(self.get_base_layer().qweight.device)
        else:
            self.to(self.get_base_layer().weight.device)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_weights):
        if adapter_name in self.lora_Ut.keys():
            ## initialize s
            if init_weights in ["s_kunif", "suv_kunif"]:
                nn.init.kaiming_uniform_(self.lora_S[adapter_name])
            else:
                nn.init.zeros_(self.lora_S[adapter_name])

            ## initialize u and v
            if init_weights in ["uv_kunif", "suv_kunif"]:
                nn.init.kaiming_uniform_(self.lora_V[adapter_name])
                nn.init.kaiming_uniform_(self.lora_Ut[adapter_name])
            else:
                if hasattr(self.get_base_layer(), "qweight"):
                    # QuantLinear
                    warnings.warn("SVD of quantized layer might be undefined.")
                    v, _, ut = torch.linalg.svd(self.get_base_layer().qweight.T, full_matrices=False)
                else:
                    v, _, ut = torch.linalg.svd(self.get_base_layer().weight.T, full_matrices=False)

                if self.r[adapter_name] > min(ut.shape[1], v.shape[0]):
                    self.lora_V[adapter_name].data = v.contiguous()
                    self.lora_Ut[adapter_name].data = ut.contiguous()
                else:
                    self.lora_V[adapter_name].data = v[:, : self.r[adapter_name]].contiguous()
                    self.lora_Ut[adapter_name].data = ut[: self.r[adapter_name], :].contiguous()

            if adapter_name in self.lora_rank_one_At.keys():
                nn.init.kaiming_uniform_(self.lora_rank_one_B[adapter_name])
                nn.init.kaiming_uniform_(self.lora_rank_one_At[adapter_name])

            if adapter_name in self.lora_gate_S.keys():
                nn.init.zeros_(self.lora_gate_S[adapter_name])


class SVDLinear(nn.Module, SVFTLayer):
    # SVD-based adaptation by a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 1,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_weights: str = "svd",
        train_A: bool = False,
        train_B: bool = False,
        s_gating: bool = False,
        rank_one: bool = False,
        rank_one_gating: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        SVFTLayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_weights,
            train_A=train_A,
            train_B=train_B,
            s_gating=s_gating,
            rank_one=rank_one,
            rank_one_gating=rank_one_gating,
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
            if active_adapter in self.lora_Ut.keys():
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
            if active_adapter in self.lora_Ut.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        lora_Ut = self.lora_Ut[adapter]
        lora_V = self.lora_V[adapter]
        lora_S = self.lora_S[adapter]
        if adapter in self.lora_gate_S.keys():
            lora_S = F.tanh(self.lora_gate_S[adapter]) * lora_S
        w = lora_V @ (lora_Ut * lora_S)

        if adapter in self.lora_rank_one_At.keys():
            lora_rank_one = self.lora_rank_one_B[adapter] @ self.lora_rank_one_At[adapter]
            if adapter in self.lora_gate_rank_one.keys():
                lora_rank_one = F.tanh(self.lora_gate_rank_one[adapter]) * lora_rank_one
            w += lora_rank_one

        scaling = self.scaling[adapter]
        return transpose(w, self.fan_in_fan_out) * scaling

    def get_delta_weight_transpose(self, adapter) -> torch.Tensor:
        lora_Ut = self.lora_Ut[adapter]
        lora_V = self.lora_V[adapter]
        lora_S = self.lora_S[adapter]
        if adapter in self.lora_gate_S.keys():
            lora_S = F.tanh(self.lora_gate_S[adapter]) * lora_S
        w = lora_V @ (lora_Ut * lora_S)

        if adapter in self.lora_rank_one_At.keys():
            lora_rank_one = self.lora_rank_one_B[adapter] @ self.lora_rank_one_At[adapter]
            if adapter in self.lora_gate_rank_one.keys():
                lora_rank_one = F.tanh(self.lora_gate_rank_one[adapter]) * lora_rank_one
            w += lora_rank_one

        scaling = self.scaling[adapter]
        return w * scaling

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
                if active_adapter not in self.lora_Ut.keys():
                    continue

                dropout = self.lora_dropout[active_adapter]
                x = x.to(self.lora_Ut[active_adapter].dtype)
                w = self.get_delta_weight_transpose(active_adapter)
                result += dropout(x) @ w

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "svft." + rep
