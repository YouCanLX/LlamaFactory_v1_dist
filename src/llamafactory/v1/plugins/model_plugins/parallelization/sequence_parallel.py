# Copyright 2025 the LlamaFactory team.
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

import os
import sys
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers

from ....accelerator.interface import Dim, DistributedInterface
from ....utils import logging
from ....utils.plugin import BasePlugin
from ....utils.types import ModelOutput
from .ulysses import (
    UlyssesAttention,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    set_ulysses_sequence_parallel_group,
)


logger = logging.get_logger(__name__)


def _cross_entropy_none_chunked(
    shift_logits: torch.Tensor, shift_labels: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Cross-entropy with reduction='none', chunked along the token dimension.

    At 70k ctx + CP the per-rank token count is still huge; a single
    F.cross_entropy(..., reduction='none') can reserve ~20 GiB for the softmax
    workspace. Chunking caps peak memory without changing numerics. Keep the
    full logits tensor in its model dtype and cast each slice to fp32 here, so
    we do not materialize the entire [tokens, vocab] matrix in fp32.
    Override chunk size via LLAMAFACTORY_V1_CE_CHUNK_SIZE (0 disables chunking).
    """
    num_tokens = shift_logits.size(0)
    chunk_size = int(os.environ.get("LLAMAFACTORY_V1_CE_CHUNK_SIZE", "4096"))
    if chunk_size <= 0 or num_tokens <= chunk_size:
        return -F.cross_entropy(shift_logits.float(), shift_labels, reduction="none").view(batch_size, -1)

    chunks: list[torch.Tensor] = []
    start = 0
    min_chunk_size = int(os.environ.get("LLAMAFACTORY_V1_CE_MIN_CHUNK_SIZE", "512"))
    while start < num_tokens:
        end = min(start + chunk_size, num_tokens)
        try:
            chunks.append(
                -F.cross_entropy(shift_logits[start:end].float(), shift_labels[start:end], reduction="none")
            )
            start = end
        except torch.OutOfMemoryError:
            if chunk_size <= min_chunk_size:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(min_chunk_size, chunk_size // 2)
            logger.warning_rank0(f"Reducing sequence-parallel CE chunk size to {chunk_size} after CUDA OOM.")
    return torch.cat(chunks).view(batch_size, -1)


class SequenceParallelModelPlugin(BasePlugin):
    def __call__(self, model, model_args):
        return super().__call__(model, model_args)


class SequenceParallelLossPlugin(BasePlugin):
    def __call__(self, model, inputs, *args, **kwargs):
        return super().__call__(model, inputs, *args, **kwargs)


def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    sequence_parallel_size=1,
    dropout=0,
    deterministic=False,
    is_causal=True,
    group=None,
    mode="ulysses",
    attn_fn=None,
    target_dtype=None,
    **kwargs,
):
    if mode == "ulysses":
        dist_attn = UlyssesAttention(sequence_process_group=group, attn_fn=attn_fn)
        attn_output = dist_attn(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length=query_states.shape[1] * sequence_parallel_size,
            deterministic=deterministic,
            dropout_p=dropout,
            causal=is_causal,
            position_ids=kwargs.get("position_ids", None),
            target_dtype=target_dtype,
        )
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output


@SequenceParallelModelPlugin("ulysses").register()
def apply_sequence_parallel(model, model_args):
    # Replace _flash_attention_forward with new_flash_attn_forward
    module = sys.modules[model.__module__]
    cp_size = model_args.get("cp_size", 1)

    set_ulysses_sequence_parallel_group(DistributedInterface().get_group(Dim.CP))

    try:
        num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_attention_heads
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,
            model.config.text_config.num_key_value_heads,
        )

    assert num_attention_heads % cp_size == 0, "num_attention_heads must be divisible by cp_size"
    assert num_key_value_heads % cp_size == 0 or cp_size % num_key_value_heads == 0, (
        "num_key_value_heads must be divisible by cp_size"
    )

    origin_attn = transformers.modeling_flash_attention_utils._flash_attention_forward
    new_flash_attention_forward = partial(
        new_flash_attn_forward,
        group=get_ulysses_sequence_parallel_group(),
        mode="ulysses",
        attn_fn=origin_attn,
        sequence_parallel_size=cp_size,
    )

    for module_name, module in list(sys.modules.items()):
        try:
            if (
                hasattr(module, "__file__")
                and "transformers" in module.__file__
                and getattr(module._flash_attention_forward, "__name__", "") == "_flash_attention_forward"
            ):
                module._flash_attention_forward = new_flash_attention_forward
                logger.info_rank0(
                    f"Replaced _flash_attention_forward in module {module_name} with new_flash_attn_forward for sequence parallel."
                )
        except (AttributeError, TypeError):
            continue


def padding_and_split_data(data, device_mesh=None):
    if device_mesh is not None:
        cp_size = device_mesh["cp"].size()
        cp_rank = device_mesh["cp"].get_local_rank()
        cp_group = device_mesh["cp"].get_group()
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.ndim > 1:
                data_len = torch.tensor(v.shape[-1], device=v.device, dtype=torch.int64)
                global_data_len = [torch.empty_like(data_len) for _ in range(cp_size)]
                dist.all_gather(global_data_len, data_len, group=cp_group)
                max_data_len = max(global_data_len)
                pad_size = max_data_len - v.shape[-1] + (cp_size - max_data_len % cp_size) % cp_size
                if k == "labels":
                    pad_value = -100
                elif k == "loss_weights":
                    pad_value = 0.0
                else:
                    pad_value = 0
                pad_data = F.pad(v, (0, pad_size), value=pad_value)
                data[k] = torch.chunk(pad_data, chunks=cp_size, dim=-1)[cp_rank].contiguous()
    return data


@SequenceParallelLossPlugin("sequence_parallel_loss").register()
def sequence_parallel_loss(model, model_inputs):
    dist_interface = DistributedInterface()
    device_mesh = dist_interface.get_device_mesh(Dim.CP)
    # Use local rank (not global rank) as the device ordinal: on multi-node setups
    # `dist.get_rank()` returns the global rank, which exceeds the per-node GPU
    # count and triggers "CUDA error: invalid device ordinal" on worker nodes.
    device = dist_interface.current_device

    model_inputs = {
        k: v.to(device, non_blocking=True) for k, v in model_inputs.items() if isinstance(v, torch.Tensor)
    }

    model_inputs = padding_and_split_data(model_inputs, device_mesh)

    batch_size, _ = model_inputs["labels"].shape

    outputs: ModelOutput = model(**model_inputs)

    logits = outputs.logits

    labels = model_inputs["labels"]

    cp_group = get_ulysses_sequence_parallel_group()
    cp_world_size = get_ulysses_sequence_parallel_world_size(cp_group)
    cp_rank = get_ulysses_sequence_parallel_rank(cp_group)

    # use all_gather to collect labels from all sequence parallel processes
    global_labels = [torch.empty_like(labels) for _ in range(cp_world_size)]
    dist.all_gather(global_labels, labels, group=cp_group)
    labels = torch.cat(global_labels, dim=1).contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = F.pad(shift_labels, (0, 1), value=-100)
    shift_labels = torch.chunk(shift_labels, chunks=cp_world_size, dim=1)[cp_rank].contiguous()
    del global_labels, labels

    # use all_gather to collect loss_weights from all sequence parallel processes
    loss_weights = model_inputs["loss_weights"]
    global_loss_weights = [torch.empty_like(loss_weights) for _ in range(cp_world_size)]
    dist.all_gather(global_loss_weights, loss_weights, group=cp_group)
    shift_loss_weights = torch.cat(global_loss_weights, dim=1).contiguous()
    shift_loss_weights = shift_loss_weights[..., 1:].contiguous()
    del global_loss_weights, loss_weights

    shift_logits = logits.view(-1, logits.size(-1)).contiguous()
    shift_labels = shift_labels.view(-1).contiguous()

    # use all_gather to collect log_probs from all sequence parallel processes
    log_probs = _cross_entropy_none_chunked(shift_logits, shift_labels, batch_size)
    global_log_probs = dist.nn.all_gather(log_probs, group=cp_group)
    global_log_probs = torch.cat(global_log_probs, dim=1).contiguous()
    log_probs = global_log_probs[..., :-1].contiguous()

    loss = (-log_probs * shift_loss_weights).sum() / (shift_loss_weights.sum() + 1e-6)

    return loss
