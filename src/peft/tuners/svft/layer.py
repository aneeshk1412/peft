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
from typing import List, Optional

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose


class SVFTLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ()
    other_param_names = ("svft_U", "svft_Vt")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.fan_in_fan_out = kwargs.get("fan_in_fan_out", False)

        self.r = {}
        self.use_rank_one = {}
        self.only_rank_one = {}
        self.use_gaussian = {}
        self.svft_dropout = nn.ModuleDict({})

        # For SVD singular values
        self.svft_lambda_s = nn.ParameterDict({})
        self.svft_std = nn.ParameterDict({})
        self.svft_gaussian = {}
        self.svft_A = nn.ParameterDict({})
        self.svft_B = nn.ParameterDict({})

        # SVD vectors
        self.svft_U = nn.ParameterDict({})
        self.svft_Vt = nn.ParameterDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name,
        r,
        use_gaussian,
        use_rank_one,
        only_rank_one,
        svft_dropout,
        init_weights,
    ):
        if r is None:
            r = min(self.in_features, self.out_features)
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r

        self.use_gaussian[adapter_name] = use_gaussian
        self.use_rank_one[adapter_name] = use_rank_one
        self.only_rank_one[adapter_name] = only_rank_one

        if svft_dropout > 0.0:
            dropout_layer = nn.Dropout(p=svft_dropout)
        else:
            dropout_layer = nn.Identity()
        self.svft_dropout[adapter_name] = dropout_layer

        # Actual trainable parameters
        self.svft_lambda_s[adapter_name] = nn.Parameter(torch.zeros(1, r), requires_grad=not only_rank_one)
        if not only_rank_one:
            self.adapter_layer_names += ("svft_lambda_s",)

        self.svft_std[adapter_name] = nn.Parameter(torch.zeros(()), requires_grad=use_gaussian)
        if use_gaussian:
            self.adapter_layer_names += ("svft_std",)

        self.svft_B[adapter_name] = nn.Parameter(torch.zeros(self.out_features, 1), requires_grad=use_rank_one)
        self.svft_A[adapter_name] = nn.Parameter(torch.zeros(1, self.in_features), requires_grad=use_rank_one)
        if use_rank_one:
            self.adapter_layer_names += ("svft_A", "svft_B")

        # Non trainable Parameters
        self.svft_U[adapter_name] = nn.Parameter(torch.zeros(self.out_features, r), requires_grad=False)
        self.svft_Vt[adapter_name] = nn.Parameter(torch.zeros(r, self.in_features), requires_grad=False)

        self.reset_svft_parameters(adapter_name, init_weights)

        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)

        self.set_adapter(self.active_adapters)

    def reset_svft_parameters(self, adapter_name, init_weights):
        if adapter_name in self.svft_lambda_s.keys():
            with torch.no_grad():
                nn.init.zeros_(self.svft_lambda_s[adapter_name])
                nn.init.uniform_(self.svft_std[adapter_name], a=0.0, b=1.0)

                nn.init.kaiming_uniform_(self.svft_B[adapter_name], a=sqrt(5))
                nn.init.zeros_(self.svft_A[adapter_name])

                if init_weights == "ortho":
                    nn.init.orthogonal_(self.svft_U[adapter_name])
                    nn.init.orthogonal_(self.svft_Vt[adapter_name])
                elif init_weights == "svd":
                    if self.fan_in_fan_out:
                        U, _, Vt = torch.linalg.svd(self.get_base_layer().weight.T, full_matrices=False)
                        self.svft_U[adapter_name].data = U[:, : self.r[adapter_name]].contiguous()
                        self.svft_Vt[adapter_name].data = Vt[: self.r[adapter_name], :].contiguous()
                    else:
                        U, _, Vt = torch.linalg.svd(self.get_base_layer().weight, full_matrices=False)
                        self.svft_U[adapter_name].data = U[:, : self.r[adapter_name]].contiguous()
                        self.svft_Vt[adapter_name].data = Vt[: self.r[adapter_name], :].contiguous()
                else:
                    raise ValueError(f"Unknown weight initialization method {init_weights}")

                base_distribution = dist.Normal(0, self.svft_std[adapter_name])
                sigmoid_transform = dist.transforms.SigmoidTransform()
                scale_shift_transform = dist.transforms.AffineTransform(loc=-1, scale=2)
                self.svft_gaussian[adapter_name] = dist.TransformedDistribution(
                    base_distribution, [sigmoid_transform, scale_shift_transform]
                )


class SVFTLinear(nn.Linear, SVFTLayer):
    # SVFT implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int,
        use_gaussian: bool = False,
        use_rank_one: bool = False,
        only_rank_one: bool = False,
        svft_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_weights: str = "svd",
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        SVFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, use_gaussian, use_rank_one, only_rank_one, svft_dropout, init_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

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
            if active_adapter in self.svft_lambda_s.keys():
                base_layer = self.get_base_layer()
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
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.svft_lambda_s.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        svft_lambda_s = self.svft_lambda_s[adapter]

        device = svft_lambda_s.device
        dtype = svft_lambda_s.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        svft_U = self.svft_U[adapter]
        svft_Vt = self.svft_Vt[adapter]
        svft_A = self.svft_A[adapter]
        svft_B = self.svft_B[adapter]

        if cast_to_fp32:
            svft_lambda_s = svft_lambda_s.float()
            svft_U = svft_U.float()
            svft_Vt = svft_Vt.float()
            svft_B = svft_B.float()
            svft_A = svft_A.float()

        if not self.use_rank_one[adapter]:
            output_tensor = transpose((svft_U * svft_lambda_s) @ svft_Vt, self.fan_in_fan_out)
        elif self.only_rank_one[adapter]:
            output_tensor = transpose(svft_B @ svft_A, self.fan_in_fan_out)
        else:
            output_tensor = transpose((svft_U * svft_lambda_s) @ svft_Vt + svft_B @ svft_A, self.fan_in_fan_out)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.svft_lambda_s[adapter].data = svft_lambda_s.to(dtype)
            self.svft_U[adapter].data = svft_U.to(dtype)
            self.svft_Vt[adapter].data = svft_Vt.to(dtype)
            self.svft_B[adapter].data = svft_B.to(dtype)
            self.svft_A[adapter].data = svft_A.to(dtype)

        return output_tensor

    def forward_adapter(self, adapter_name: str, x: torch.Tensor) -> torch.Tensor:
        svft_lambda_s = self.svft_lambda_s[adapter_name]
        svft_U = self.svft_U[adapter_name]
        svft_Vt = self.svft_Vt[adapter_name]
        svft_A = self.svft_A[adapter_name]
        svft_B = self.svft_B[adapter_name]

        if not self.use_rank_one[adapter_name]:
            return F.linear(F.linear(x, svft_Vt) * svft_lambda_s, svft_U)

        if self.only_rank_one[adapter_name]:
            return F.linear(F.linear(x, svft_A), svft_B)

        return F.linear(F.linear(x, svft_Vt) * svft_lambda_s, svft_U) + F.linear(F.linear(x, svft_A), svft_B)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.svft_lambda_s.keys():
                    continue

                dropout = self.svft_dropout[active_adapter]
                x = x.to(self.svft_lambda_s[active_adapter].dtype)
                result = result + self.forward_adapter(active_adapter, dropout(x))

        result = result.to(previous_dtype)
        return result
