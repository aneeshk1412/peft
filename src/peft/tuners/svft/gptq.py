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
import torch

from .layer import SVFTLayer


class SVDQuantLinear(torch.nn.Module, SVFTLayer):
    def __init__(
        self,
        base_layer,
        adapter_name,
        r: int = 1,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_weights: str = "svd",
        train_A: bool = False,
        train_B: bool = False,
        s_gating: bool = False,
        rank_one: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        SVFTLayer.__init__(self, base_layer)

        # self.base_layer and self.quant_linear_module are the same; we need the former for consistency and the latter
        # for backwards compatibility
        self.quant_linear_module = base_layer
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.quant_linear_module(x)

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

            # TODO: here, the dtype conversion is applied on the *whole expression*,
            # not the intermediate result, unlike for SVDLinear8bitLT and
            # SVDLinear4bit, is that correct?
            if requires_conversion:
                output = output.to(expected_dtype)
            result += output
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "svft." + rep
