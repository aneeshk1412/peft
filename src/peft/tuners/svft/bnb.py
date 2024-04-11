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

from typing import Any

import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from .layer import SVFTLayer


if is_bnb_available():

    class SVDLinear8bitLt(torch.nn.Module, SVFTLayer):
        # Low-rank matrix for SVD-based adaptation
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 1,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # note: no check for self.merged because merging is not supported (yet)
            result = self.base_layer(x)

            if self.disable_adapters:
                return result

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_S.keys():
                    continue
                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    if x.dtype != torch.float32:
                        x = x.float()

                dropout = self.lora_dropout[active_adapter]
                w = self.get_delta_weight_transpose(active_adapter)
                output = dropout(x) @ w

                if requires_conversion:
                    output = output.to(expected_dtype)
                # inplace operation on view is forbidden for MatMul8bitLtBackward, so avoid it
                result = result + output
            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "svft." + rep


if is_bnb_4bit_available():

    class SVDLinear4bit(torch.nn.Module, SVFTLayer):
        # Low-rank matrix for SVD-based adaptation
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 1,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
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

        def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            # note: no check for self.merged because merging is not supported (yet)
            result = self.base_layer(x, *args, **kwargs)

            if self.disable_adapters:
                return result

            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
            result = result.clone()

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_S.keys():
                    continue

                lora_S = self.lora_S[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    compute_dtype = lora_S.dtype
                    if x.dtype != compute_dtype:
                        x = x.to(compute_dtype)

                dropout = self.lora_dropout[active_adapter]
                w = self.get_delta_weight_transpose(active_adapter)
                output = dropout(x) @ w

                if requires_conversion:
                    output = output.to(expected_dtype)
                result += output
            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "svft." + rep
