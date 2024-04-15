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
    adapter_layer_names = ()
    # other_param_names is defined in LoraLayer

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.lora_svft_Ut = nn.ModuleDict({})
        self.lora_svft_V = nn.ModuleDict({})
        self.lora_svft_delta_S = nn.ParameterDict({})
        self.lora_svft_gate_delta_S = nn.ParameterDict({})
        self.lora_svft_rank_r_A = nn.ModuleDict({})
        self.lora_svft_rank_r_B = nn.ModuleDict({})
        self.lora_svft_gate_rank_r = nn.ParameterDict({})

        self.train_delta_S = {}
        self.gate_delta_S = {}
        self.gate_rank_r = {}
        self.rank_r = {}

        self.init_U = {}
        self.init_V = {}
        self.init_delta_S = {}
        self.elems_per_bin_delta_S = {}

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        train_delta_S=True,
        init_U="svd",
        init_V="svd",
        init_delta_S="svd",
        elems_per_bin_delta_S=1,
        gate_delta_S=False,
        rank_r=None,
        gate_rank_r=False,
    ):
        if r is None:
            r = min(self.in_features, self.out_features)

        if r <= 0:
            raise ValueError(f"`r` should be a positive integer, but the value passed is {r}")

        if r > min(self.in_features, self.out_features):
            r = min(self.in_features, self.out_features)

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else 1.0 / sqrt(r)

        self.train_delta_S[adapter_name] = train_delta_S
        self.gate_delta_S[adapter_name] = gate_delta_S
        self.gate_rank_r[adapter_name] = gate_rank_r
        self.rank_r[adapter_name] = rank_r

        self.init_U[adapter_name] = init_U
        self.init_V[adapter_name] = init_V
        self.init_delta_S[adapter_name] = init_delta_S
        self.elems_per_bin_delta_S[adapter_name] = elems_per_bin_delta_S

        self.lora_svft_Ut[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        self.lora_svft_V[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_svft_Ut[adapter_name].requires_grad_(False)
        self.lora_svft_V[adapter_name].requires_grad_(False)

        if train_delta_S:
            self.lora_svft_delta_S[adapter_name] = nn.Parameter(
                torch.zeros(1, r // elems_per_bin_delta_S), requires_grad=train_delta_S
            )
            self.adapter_layer_names = self.adapter_layer_names + ("lora_svft_delta_S",)

        if gate_delta_S:
            self.lora_svft_gate_delta_S[adapter_name] = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.adapter_layer_names = self.adapter_layer_names + ("lora_svft_gate_delta_S",)

        if rank_r:
            self.lora_svft_rank_r_A[adapter_name] = nn.Linear(self.in_features, rank_r, bias=False)
            self.lora_svft_rank_r_B[adapter_name] = nn.Linear(rank_r, self.out_features, bias=False)
            self.adapter_layer_names = self.adapter_layer_names + ("lora_svft_rank_r_A", "lora_svft_rank_r_B")

        if gate_rank_r:
            self.lora_svft_gate_rank_r[adapter_name] = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.adapter_layer_names = self.adapter_layer_names + ("lora_svft_gate_rank_r",)

        self.reset_lora_parameters(adapter_name)

        if hasattr(self.get_base_layer(), "qweight"):
            # QuantLinear
            self.to(self.get_base_layer().qweight.device)
        else:
            self.to(self.get_base_layer().weight.device)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_svft_Ut.keys():
            if "svd" in (self.init_U[adapter_name], self.init_V[adapter_name]):
                if self.fan_in_fan_out:
                    U, _, Vt = torch.linalg.svd(self.get_base_layer().weight.T, full_matrices=False)
                else:
                    U, _, Vt = torch.linalg.svd(self.get_base_layer().weight, full_matrices=False)

            if self.init_U[adapter_name] == "svd":
                self.lora_svft_Ut[adapter_name].weight.data = U[:, : self.r[adapter_name]].contiguous()
            elif self.init_U[adapter_name] == "kunif":
                nn.init.kaiming_uniform_(self.lora_svft_Ut[adapter_name].weight)

            if self.init_V[adapter_name] == "svd":
                self.lora_svft_V[adapter_name].weight.data = Vt[: self.r[adapter_name], :].contiguous()
            elif self.init_V[adapter_name] == "kunif":
                nn.init.kaiming_uniform_(self.lora_svft_V[adapter_name].weight)

            if self.init_delta_S[adapter_name] == "svd":
                nn.init.zeros_(self.lora_svft_delta_S[adapter_name])
            elif self.init_delta_S[adapter_name] == "kunif":
                nn.init.kaiming_uniform_(self.lora_svft_delta_S[adapter_name])

            if self.gate_delta_S[adapter_name]:
                nn.init.zeros_(self.lora_svft_gate_delta_S[adapter_name])

            if self.rank_r[adapter_name]:
                nn.init.zeros_(self.lora_svft_rank_r_A[adapter_name].weight)
                nn.init.kaiming_uniform_(self.lora_svft_rank_r_B[adapter_name].weight)

            if self.gate_rank_r[adapter_name]:
                nn.init.zeros_(self.lora_svft_gate_rank_r[adapter_name])
                nn.init.kaiming_uniform_(self.lora_svft_rank_r_A[adapter_name].weight)


class SVDLinear(nn.Module, SVFTLayer):
    # SVD-based adaptation by a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r=None,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        train_delta_S=True,
        init_U="svd",
        init_V="svd",
        init_delta_S="svd",
        elems_per_bin_delta_S=1,
        gate_delta_S=False,
        rank_r=None,
        gate_rank_r=False,
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
            train_delta_S=train_delta_S,
            init_U=init_U,
            init_V=init_V,
            init_delta_S=init_delta_S,
            elems_per_bin_delta_S=elems_per_bin_delta_S,
            gate_delta_S=gate_delta_S,
            rank_r=rank_r,
            gate_rank_r=gate_rank_r,
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
            if active_adapter in self.lora_svft_Ut.keys():
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
            if active_adapter in self.lora_svft_Ut.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        Ut = self.lora_svft_Ut[adapter].weight
        V = self.lora_svft_V[adapter].weight
        delta_S = self.lora_svft_delta_S[adapter]
        scaling = self.scaling[adapter]
        elems_per_bin_delta_S = self.elems_per_bin_delta_S[adapter]
        delta_S = F.sigmoid(self.lora_svft_gate_delta_S[adapter]) * delta_S

        w = Ut @ (delta_S.repeat_interleave(elems_per_bin_delta_S, dim=-1).T * V)
        return transpose(w, self.fan_in_fan_out) * scaling

    def forward_adapter(self, adapter_name: str, x: torch.Tensor) -> torch.Tensor:
        Ut = self.lora_svft_Ut[adapter_name]
        V = self.lora_svft_V[adapter_name]
        elems_per_bin_delta_S = self.elems_per_bin_delta_S[adapter_name]
        delta_S = self.lora_svft_delta_S[adapter_name]
        delta_S = F.sigmoid(self.lora_svft_gate_delta_S[adapter_name]) * delta_S
        return Ut(delta_S.repeat_interleave(elems_per_bin_delta_S, dim=-1) * V(x)) * self.scaling[adapter_name]

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
                if active_adapter not in self.lora_svft_Ut.keys():
                    continue

                dropout = self.lora_dropout[active_adapter]
                x = x.to(self.lora_svft_Ut[active_adapter].weight.dtype)
                result += self.forward_adapter(active_adapter, dropout(x))

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "svft." + rep
