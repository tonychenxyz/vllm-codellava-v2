# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen3 model compatible with HuggingFace weights."""

import base64
import io
from collections.abc import Iterable, Mapping
from typing import Any, Optional, Sequence, Union

import torch
from torch import nn
from transformers import BatchFeature, Qwen3Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (ModalityData, MultiModalDataDict,
                                    MultiModalFieldConfig)
from vllm.multimodal.parse import (EmbeddingItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

from .interfaces import (SupportsEagle3, SupportsLoRA, SupportsMultiModal,
                         SupportsPP)
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    maybe_prefix, merge_multimodal_embeddings)

logger = init_logger(__name__)


def _load_tensor_from_base64(data: str) -> torch.Tensor:
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Failed to decode base64 embedding payload") from exc

    buffer = io.BytesIO(raw)
    tensor = torch.load(buffer, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Embedding payload must serialize a tensor")
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    return tensor


class Qwen3VisionEmbeddingParser(MultiModalDataParser):

    @staticmethod
    def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected embedding tensor with 2 dimensions, got {tensor.shape}"
            )
        return tensor.contiguous()

    @staticmethod
    def _to_tensor(payload: Union[str, dict[str, Any], list[Any], torch.Tensor]
                   ) -> torch.Tensor:
        if isinstance(payload, torch.Tensor):
            tensor = payload
        elif isinstance(payload, str):
            tensor = _load_tensor_from_base64(payload)
        elif isinstance(payload, dict):
            if "data" not in payload:
                raise ValueError(
                    "Embedding dict payload must contain a 'data' field")
            tensor = _load_tensor_from_base64(str(payload["data"]))
        else:
            tensor = torch.tensor(payload, dtype=torch.float32)

        if tensor.device != torch.device("cpu"):
            tensor = tensor.cpu()
        if tensor.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            tensor = tensor.float()

        return Qwen3VisionEmbeddingParser._ensure_2d(tensor.float())

    def _parse_vision_embedding(
        self, data: ModalityData[Any]
    ) -> Optional[EmbeddingItems]:
        if data is None:
            return None
        payloads: list[Any]
        if isinstance(data, list):
            payloads = data
        else:
            payloads = [data]

        tensors = [self._to_tensor(payload) for payload in payloads]
        return EmbeddingItems(tensors, "vision_embedding")

    def _get_subparsers(self):
        subparsers = super()._get_subparsers()
        subparsers["vision_embedding"] = self._parse_vision_embedding
        return subparsers


class Qwen3VisionProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> Qwen3Config:
        return self.ctx.get_hf_config(Qwen3Config)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"vision_embedding": 1}


class Qwen3VisionDummyInputsBuilder(
        BaseDummyInputsBuilder[Qwen3VisionProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        count = mm_counts.get("vision_embedding", 0)
        return "<|fim_pad|>" * count

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        count = mm_counts.get("vision_embedding", 0)
        if count == 0:
            return {}

        hidden_size = self.info.get_hf_config().hidden_size
        dummy = torch.zeros((1, hidden_size), dtype=torch.float32)
        return {"vision_embedding": [dummy.clone() for _ in range(count)]}


class Qwen3VisionMultiModalProcessor(
        BaseMultiModalProcessor[Qwen3VisionProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        return Qwen3VisionEmbeddingParser()

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: Mapping[str, Any],
    ) -> Sequence[PromptReplacement]:
        tokenizer = self.info.get_tokenizer()
        placeholder_token = "<|fim_pad|>"
        placeholder_token_id = tokenizer.convert_tokens_to_ids(
            placeholder_token)
        if placeholder_token_id is None or placeholder_token_id < 0:
            raise ValueError(
                "Unable to determine token id for <|fim_pad|> placeholder"
            )

        def build_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
            embeds = mm_items["vision_embedding"].get(item_idx)
            num_tokens = embeds.shape[0]
            tokens = [placeholder_token_id] * num_tokens
            return PromptUpdateDetails.from_seq(tokens)

        return [
            PromptReplacement(
                modality="vision_embedding",
                target=[placeholder_token_id],
                replacement=lambda idx, fn=build_replacement: fn(idx),
            )
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, Any]:
        sizes = hf_inputs.get("vision_embedding_sizes")
        if sizes is None:
            return {}

        if not isinstance(sizes, torch.Tensor):
            sizes_tensor = torch.tensor(sizes, dtype=torch.long)
        else:
            sizes_tensor = sizes.to(dtype=torch.long, device="cpu")

        if sizes_tensor.numel() == 0:
            hf_inputs.pop("vision_embedding_sizes", None)
            return {}

        field_config = MultiModalFieldConfig.flat_from_sizes(
            "vision_embedding",
            sizes_tensor,
        )
        hf_inputs.pop("vision_embedding_sizes", None)
        return {"vision_embedding_embeds": field_config}

    def _apply_hf_processor_text_mm(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> tuple[list[int], BatchFeature, bool]:
        if all(isinstance(items, EmbeddingItems) for items in mm_items.values()):
            embedding_items = mm_items.get("vision_embedding")
            if embedding_items is None:
                return super()._apply_hf_processor_text_mm(
                    prompt_text,
                    mm_items,
                    hf_processor_mm_kwargs,
                    tokenization_kwargs,
                )

            tokenizer = self.info.get_tokenizer()
            tokenized = tokenizer(prompt_text,
                                  add_special_tokens=False)  # type: ignore[call-arg]
            if isinstance(tokenized, Mapping):
                prompt_ids = tokenized.get("input_ids", [])
            else:
                prompt_ids = tokenized

            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]

            prompt_ids = list(map(int, prompt_ids)) if prompt_ids else []

            num_items = embedding_items.get_count()
            if num_items == 0:
                hidden = self.info.get_hf_config().hidden_size
                concat_embeds = torch.empty((0, hidden), dtype=torch.float32)
                sizes = torch.tensor([], dtype=torch.long)
            else:
                tensors = [
                    embedding_items.get(idx).detach().to(device="cpu")
                    for idx in range(num_items)
                ]
                sizes = torch.tensor([tensor.shape[0] for tensor in tensors],
                                      dtype=torch.long)
                concat_embeds = torch.cat(tensors, dim=0)

            processed_data = BatchFeature({
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
                "vision_embedding_embeds": concat_embeds,
                "vision_embedding_sizes": sizes,
            })

            return prompt_ids, processed_data, False

        return super()._apply_hf_processor_text_mm(
            prompt_text,
            mm_items,
            hf_processor_mm_kwargs,
            tokenization_kwargs,
        )

class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        rope_scaling: tuple | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )


@MULTIMODAL_REGISTRY.register_processor(Qwen3VisionMultiModalProcessor,
                                        info=Qwen3VisionProcessingInfo,
                                        dummy_inputs=Qwen3VisionDummyInputsBuilder)
class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3,
                       SupportsMultiModal):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self._vision_placeholder_token_id = getattr(
            config, "vision_start_token_id", None)
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self._fallback_placeholder_token_id = tokenizer.convert_tokens_to_ids(
            "<|fim_pad|>")

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality == "vision_embedding":
            return "<|fim_pad|>"
        return None

    def get_language_model(self) -> nn.Module:
        return self.model

    def _get_vision_placeholder_token_id(self) -> int:
        if self._vision_placeholder_token_id is not None:
            return self._vision_placeholder_token_id

        if (self._fallback_placeholder_token_id is not None
                and self._fallback_placeholder_token_id >= 0):
            return self._fallback_placeholder_token_id

        raise ValueError(
            "Unable to determine placeholder token id for vision embeddings")

    def _normalize_mm_embeddings(
        self, vision_embedding_embeds: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> tuple[torch.Tensor, ...]:
        def _flatten(value: Union[torch.Tensor, Sequence[torch.Tensor]]
                    ) -> list[torch.Tensor]:
            if isinstance(value, torch.Tensor):
                if value.ndim == 2:
                    return [value]
                if value.ndim == 3:
                    return [v for v in value]
                raise ValueError(
                    f"Expected 2D/3D tensor embeddings, got shape {value.shape}")
            flattened: list[torch.Tensor] = []
            for item in value:
                flattened.extend(_flatten(item))
            return flattened

        tensors = _flatten(vision_embedding_embeds)
        if not tensors:
            raise ValueError("No vision embeddings provided")

        return tuple(tensors)

    def get_multimodal_embeddings(
        self,
        *,
        vision_embedding_embeds: Union[torch.Tensor, Sequence[torch.Tensor]],
        **_: object,
    ) -> tuple[torch.Tensor, ...]:
        tensors = self._normalize_mm_embeddings(vision_embedding_embeds)
        dtype = self.model.embed_tokens.weight.dtype
        device = self.model.embed_tokens.weight.device

        processed: list[torch.Tensor] = []
        for tensor in tensors:
            tensor = tensor.to(device=device, dtype=dtype)
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 2:
                raise ValueError(
                    f"Expected 2D tensor per vision embedding, got {tensor.shape}")
            processed.append(tensor.contiguous())

        return tuple(processed)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[Sequence[torch.Tensor]] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> torch.Tensor:
        _ = attn_metadata  # unused; kept for v0 compatibility
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings:
            placeholder_id = self._get_vision_placeholder_token_id()
            mm_embeddings = tuple(multimodal_embeddings)
            logger.warning(
                "Qwen3 vision embeddings merged: num_items=%d, shapes=%s",  # noqa: E501
                len(mm_embeddings),
                [tuple(t.shape) for t in mm_embeddings],
            )
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                mm_embeddings,
                placeholder_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
