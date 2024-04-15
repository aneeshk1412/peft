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

import torch
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.lora import LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_SVFT_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    get_auto_gptq_quant_linear,
    get_quantization_config,
)

from .config import SVFTConfig
from .gptq import SVDQuantLinear
from .layer import SVDLinear, SVFTLayer


class SVFTModel(LoraModel):
    """
    Creates SVFT model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`SVFTConfig`]): The configuration of the SVFT model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The SVFT model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import SVFTModel, SVFTConfig
        >>> config = SVFTConfig(
                peft_type="SVFT", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
                lora_dropout=0.01, train_U=False, train_V=False, train_delta_S=True,
                init_U="svd", init_V="svd", init_delta_S="zero", elems_per_bin_delta_S=1,
                gate_delta_S=False, rank_r=None, gate_rank_r=False,
                fan_in_fan_out=True, inference_mode=False
            )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> model = SVFTModel(model, config, "default")

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`SVFTConfig`]): The configuration of the SVFT model.
    """

    # Note: don't redefine prefix here, it should be inherited from LoraModel

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

        traininable_mode_counter = 0
        for config in self.peft_config.values():
            if not config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "SVFTModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one you want to train."
            )

        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name

    def _check_new_adapter_config(self, config: SVFTConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        super()._check_new_adapter_config(config)

        traininable_mode_counter = 0
        for config_ in self.peft_config.values():
            if not config_.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one "
                "you want to train."
            )

    def _create_and_replace(
        self,
        svft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        kwargs = {
            "r": svft_config.r,
            "lora_alpha": svft_config.lora_alpha,
            "lora_dropout": svft_config.lora_dropout,
            "fan_in_fan_out": svft_config.fan_in_fan_out,
            "init_lora_weights": svft_config.init_lora_weights,
            "train_delta_S": svft_config.train_delta_S,
            "init_U": svft_config.init_U,
            "init_V": svft_config.init_V,
            "init_delta_S": svft_config.init_delta_S,
            "elems_per_bin_delta_S": svft_config.elems_per_bin_delta_S,
            "gate_delta_S": svft_config.gate_delta_S,
            "rank_r": svft_config.rank_r,
            "gate_rank_r": svft_config.gate_rank_r,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        if (kwargs["loaded_in_8bit"] or kwargs["loaded_in_4bit"]) and not is_bnb_available():
            raise ImportError(
                "To use SVFT with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

        quantization_config = get_quantization_config(self.model, method="gptq")
        if quantization_config is not None:
            kwargs["gptq_quantization_config"] = quantization_config

        # If it is not an SVFTLayer, create a new module, else update it with new adapters
        if not isinstance(target, SVFTLayer):
            new_module = self._create_new_module(svft_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                svft_config.r,
                svft_config.lora_alpha,
                svft_config.lora_dropout,
                train_delta_S=svft_config.train_delta_S,
                init_U=svft_config.init_U,
                init_V=svft_config.init_V,
                init_delta_S=svft_config.init_delta_S,
                elems_per_bin_delta_S=svft_config.elems_per_bin_delta_S,
                gate_delta_S=svft_config.gate_delta_S,
                rank_r=svft_config.rank_r,
                gate_rank_r=svft_config.gate_rank_r,
            )

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # avoid eager bnb import
        if is_bnb_available():
            import bitsandbytes as bnb

            from .bnb import SVDLinear8bitLt
        if is_bnb_4bit_available():
            from .bnb import SVDLinear4bit

        gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            kwargs.update(
                {
                    "has_fp16_weights": target_base_layer.state.has_fp16_weights,
                    "memory_efficient_backward": target_base_layer.state.memory_efficient_backward,
                    "threshold": target_base_layer.state.threshold,
                    "index": target_base_layer.index,
                }
            )
            new_module = SVDLinear8bitLt(target, adapter_name, **kwargs)
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            new_module = SVDLinear4bit(target, adapter_name, **fourbit_kwargs)
        elif AutoGPTQQuantLinear is not None and isinstance(target, AutoGPTQQuantLinear):
            new_module = SVDQuantLinear(target, adapter_name, **kwargs)
        else:
            if isinstance(target_base_layer, torch.nn.Linear):
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target_base_layer, Conv1D):
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = SVDLinear(target, adapter_name, **kwargs)

        return new_module

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_SVFT_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_SVFT_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)
