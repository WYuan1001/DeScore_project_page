# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss, compute_bt_loss, compute_multi_bt_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig
from tensordict import TensorDict
import numpy as np

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass

__all__ = ["DataParallelPPOActor", "DataParallelPairPPOActor"]

import traceback
import sys

def excepthook(exc_type, exc_value, exc_tb):
    rank = dist.get_rank() if dist.is_initialized() else -1
    print(f'[Rank {rank}] Exception caught:', flush=True)
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.stdout.flush()

sys.excepthook = excepthook


def process_to_pair_per(object, rollout_batch_size=8, n_split=False): # n_split指的就是要不要把rollout变成一个单独的纬度，变成batch_size*n*2*dim
    object_ = [object[i:i+rollout_batch_size] for i in range(0, len(object), rollout_batch_size)]
    object_ = [torch.stack([object_[i], object_[i+1]], dim=1) if torch.is_tensor(object_[i]) else np.stack([object_[i], object_[i+1]], axis=1) for i in range(0, len(object_), 2)]
    if not n_split:
        process_object = torch.concat(object_, dim=0) if torch.is_tensor(object_[0]) else np.concatenate(object_, axis=0)
    else:
        process_object = torch.stack(object_, dim=0) if torch.is_tensor(object_[0]) else np.stack(object_, axis=0)
    return process_object

def process_to_pair(batch, rollout_batch_size=8, n_split=False):
    non_tensor_key = list(batch.non_tensor_batch.keys())
    batch_size = len(batch.batch['input_ids'])
    paired  = {k: process_to_pair_per(v, rollout_batch_size, n_split) for k, v in batch.batch.items()}
    if n_split:
        batch.batch = TensorDict(paired, batch_size=([int(batch_size//(rollout_batch_size*2))]))
    else:
        batch.batch = TensorDict(paired, batch_size=([int(batch_size//2)]))
    
    for key in non_tensor_key:
        batch.non_tensor_batch[key] = process_to_pair_per(batch.non_tensor_batch[key], rollout_batch_size, n_split)
    return batch

def process_pair_to_seq(batch, n_split=False):
    if n_split:
        batch_size = batch.batch['input_ids'].size()[0]*batch.batch['input_ids'].size()[1]
        paired  = {k: v.view((-1,)+v.size()[3:]) if len(v.size())>=4 else v.unsqueeze(2).repeat(1, 1, 2, 1).view((-1,v.size()[-1])) for k, v in batch.batch.items()}
        batch.batch = TensorDict(paired, batch_size=([batch_size*2]))
    else:
        batch_size = len(batch.batch['input_ids'])
        paired  = {k: v.view((-1,)+v.size()[2:]) if len(v.size())>=3 else v.unsqueeze(1).repeat(1, 2, 1).view((-1,v.size()[-1])) for k, v in batch.batch.items()}
        batch.batch = TensorDict(paired, batch_size=([batch_size*2]))
    
    for key, value in batch.non_tensor_batch.items():
        batch.non_tensor_batch[key] = value.reshape(-1)
    return batch

class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        head_module: nn.Module, 
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        trainable_head: bool=False
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.head_module = head_module
        self.actor_optimizer = actor_optimizer
        self.trainable_head = trainable_head
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits
    
    def compute_head_reward(self, data: DataProto, trainable=False) -> torch.Tensor:
        if not trainable: 
            self.actor_module.eval()
            self.head_module.eval()
            with torch.no_grad():
                special_hiddens = self._compute_special(data)
                reward = self.head_module(special_hiddens)
        else:
            self.actor_module.train()
            self.head_module.train()
            special_hiddens = self._compute_special(data)
            reward = self.head_module(special_hiddens)
        return reward
    
    def _compute_special(self, data: DataProto) -> torch.Tensor:
        try:
            temperature = data.meta_info["temperature"]
        except:
            temperature = 0.0
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]
        self.special_token_ids = data.meta_info['special_token_ids']

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        special_hidden_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute special token hidden states", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            special_hidden_state = self._cal_micro_batch_special(model_inputs)
            special_hidden_lst.append(special_hidden_state)

        special_hiddens = torch.concat(special_hidden_lst, dim=0)

        if self.config.dynamic_batching:
            special_hiddens = restore_dynamic_batch(special_hiddens, batch_idx_list)
        
        return special_hiddens
    
    def _cal_micro_batch_special(self, micro_batch: dict[str, torch.Tensor], only_bt=False) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            print(f'Inference, return hidden_states for special token (padding_free mode)...')
            
            if not hasattr(self, 'special_token_ids') or self.special_token_ids is None:
                raise ValueError("special_token_ids is not set. Please set it before calling with return_special=True")
            
            self.special_token_ids = [self.special_token_ids] if not isinstance(self.special_token_ids, list) else self.special_token_ids
            
            hidden_states_cache = {}
            
            def hook_fn(module, input, output):
                if hasattr(output, 'last_hidden_state'):
                    print(f'hasattr last_hidden_state')
                    hidden_states_cache['last_hidden_states'] = output.last_hidden_state # TODO: 这里不detach会不会有什么问题
                else:
                    print(f'output[0], {output}')
                    hidden_states_cache['last_hidden_states'] = output[0]
            
            target_layer = None
            
            if target_layer is None and hasattr(self.actor_module, '_fsdp_wrapped_module'):
                fsdp_module = self.actor_module._fsdp_wrapped_module
                if hasattr(fsdp_module, 'model'):
                    target_layer = fsdp_module.model
                    print(f"Found target: _fsdp_wrapped_module.model")
            
            handle = target_layer.register_forward_hook(hook_fn)
            
            try:
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    output_hidden_states=True,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                if 'last_hidden_states' not in hidden_states_cache:
                    raise ValueError("Hook did not capture hidden states")
                
                last_hidden_states_rmpad = hidden_states_cache['last_hidden_states']  # (1, total_nnz, hidden_dim) or (total_nnz, hidden_dim)
                print(f"Captured hidden states shape: {last_hidden_states_rmpad.shape}")
                
                if only_bt:
                    self._fsdp_output_anchor = output.logits.sum() * 0.0 

                if last_hidden_states_rmpad.dim() == 3:
                    last_hidden_states_rmpad = last_hidden_states_rmpad.squeeze(0)
                
            finally:
                handle.remove()

            special_token_mask_rmpad = torch.zeros_like(input_ids_rmpad.squeeze(0), dtype=torch.bool)  # (total_nnz,)
            for special_token_id in self.special_token_ids:
                special_token_mask_rmpad = special_token_mask_rmpad | (input_ids_rmpad.squeeze(0) == special_token_id) # 之前这里用的是input_ids_rmpad
            
            masked_token_ids = input_ids_rmpad.squeeze(0)[special_token_mask_rmpad]
            assert all(tid.item() in self.special_token_ids for tid in masked_token_ids), \
                f"Mask error: got token ids {masked_token_ids.tolist()}, expected {self.special_token_ids}"
            
            num_special_total = special_token_mask_rmpad.sum().item()
            
            pooled_hidden_states = last_hidden_states_rmpad[special_token_mask_rmpad, :]  # (num_special_tokens_total, hidden_dim)
            
            num_special_per_sample = num_special_total // batch_size
            print(f"Each sample has {num_special_per_sample} special tokens")

            if num_special_per_sample>1:
                pooled_hidden_states = pooled_hidden_states.view(batch_size, num_special_per_sample, -1)
        
            print(f"Final pooled_hidden_states shape: {pooled_hidden_states.shape}")
            return pooled_hidden_states # [token_num, hidden_states]
        else:
            print(f'Inference, return hidden_states for special token (non-padding_free mode)...')
            
            if not hasattr(self, 'special_token_ids') or self.special_token_ids is None:
                raise ValueError("special_token_ids is not set. Please set it before calling with return_special=True")
            
            self.special_token_ids = [self.special_token_ids] if not isinstance(self.special_token_ids, list) else self.special_token_ids
            
            hidden_states_cache = {}
            
            def hook_fn(module, input, output):
                """Hook 函数来捕获最后一层的输出"""
                if hasattr(output, 'last_hidden_state'):
                    print(f'hasattr last_hidden_state')
                    hidden_states_cache['last_hidden_states'] = output.last_hidden_state
                else:
                    print(f'output[0], {output}')
                    hidden_states_cache['last_hidden_states'] = output[0]
            
            target_layer = None
            if target_layer is None and hasattr(self.actor_module, '_fsdp_wrapped_module'):
                fsdp_module = self.actor_module._fsdp_wrapped_module
                if hasattr(fsdp_module, 'model'):
                    target_layer = fsdp_module.model
                    print(f"Found target: _fsdp_wrapped_module.model")
            
            if target_layer is None:
                print("Could not automatically find target layer.")
                raise ValueError("Cannot find the last layer of the model.")
            
            handle = target_layer.register_forward_hook(hook_fn)
            
            try:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                
                if 'last_hidden_states' not in hidden_states_cache:
                    raise ValueError("Hook did not capture hidden states")
                
                last_hidden_states = hidden_states_cache['last_hidden_states']  # (batch_size, seqlen, hidden_dim)
                print(f"Captured hidden states shape: {last_hidden_states.shape}")
                
            finally:
                handle.remove()
            
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)  # (batch_size, seqlen)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
                
            
            num_special_total = special_token_mask.sum().item()
            
            pooled_hidden_states_list = []
            for i in range(batch_size):
                sample_hidden = last_hidden_states[i]  # (seqlen, hidden_dim)
                sample_mask = special_token_mask[i]     # (seqlen,)
                sample_special_hidden = sample_hidden[sample_mask]  # (num_special_in_sample, hidden_dim)
                pooled_hidden_states_list.append(sample_special_hidden)
                print(f"Sample {i}: found {sample_special_hidden.size(0)} special tokens")
            
            pooled_hidden_states = torch.stack(pooled_hidden_states_list, dim=0)  # (batch_size, num_special, hidden_dim)
            assert len(pooled_hidden_states) == batch_size, print(f'Exists sequence without special token ...')
            print(f"Final pooled_hidden_states shape: {pooled_hidden_states.shape}")
            return pooled_hidden_states
    

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float, return_special=False) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"] 
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz) 这个就相当于是next_token的input_ids了

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
            
            special_token_mask_rmpad = torch.zeros_like(input_ids_rmpad.squeeze(0), dtype=torch.bool)  # (total_nnz,)
            for special_token_id in self.special_token_ids:
                special_token_mask_rmpad = special_token_mask_rmpad | (input_ids_rmpad.squeeze(0) == special_token_id) # 之前这里用的是input_ids_rmpad
            assert special_token_mask_rmpad.sum().item()==0, print(f'mask sum is [{special_token_mask_rmpad.sum().item()}], there is special token when compute logit !!')
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )

            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            # 非 padding_free 模式
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)  # (batch_size, seqlen)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            assert special_token_mask.sum().item()==0, print(f'mask sum is [{special_token_mask.sum().item()}], there is special token when compute logit !!')
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )

            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)
        
        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
            if isinstance(self.head_module, FSDP) and self.trainable_head:
                self.head_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            params = list(self.actor_module.parameters()) + list(self.head_module.parameters()) if self.trainable_head else self.actor_module.parameters()
            grad_norm = nn.utils.clip_grad_norm_(params, max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        if not hasattr(self, 'special_token_ids'):
            self.special_token_ids = data.meta_info['special_token_ids']
        self.special_token_ids = [self.special_token_ids] if not isinstance(self.special_token_ids, list) else self.special_token_ids

        temperature = data.meta_info["temperature"]
        select_keys = ["log_input_ids", "log_attention_mask", "log_position_ids", "log_responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]
        try:
            data = data.select(select_keys, non_tensor_select_keys)
            
            data = data.rename("log_input_ids", "input_ids")
            data = data.rename("log_attention_mask", "attention_mask")
            data = data.rename("log_position_ids", "position_ids")
            data = data.rename("log_responses", "responses")
        except Exception as e:
            print(f'Error is {e}, current data is [{data.batch}]')
            alter_select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
            data = data.select(alter_select_keys, non_tensor_select_keys)
        
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()
        if self.trainable_head:
            self.head_module.train()
        
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys_log = ["log_input_ids", "log_attention_mask", "log_position_ids", "log_responses", "log_response_mask"]
        select_keys_special = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys_log.extend(["old_log_probs", "ref_log_probs", "advantages"])
        select_keys_special.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs", 'ground_truth', 'uid'] # TODO: 检查是不是ground_truth

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches_log = data.select(select_keys_log, non_tensor_select_keys) #.split(self.config.global_batch_size_per_device)
        mini_batches_special = data.select(select_keys_special, non_tensor_select_keys) #.split(self.config.global_batch_size_per_device) # 这里有split过，这个的split一定是整分，可以保证pair的吗

        mini_batches_log = mini_batches_log.rename("log_input_ids", "input_ids")
        mini_batches_log = mini_batches_log.rename("log_attention_mask", "attention_mask")
        mini_batches_log = mini_batches_log.rename("log_position_ids", "position_ids")
        mini_batches_log = mini_batches_log.rename("log_responses", "responses")
        mini_batches_log = mini_batches_log.rename("log_response_mask", "response_mask")

        split_size = self.config.global_batch_size_per_device // 2
        split_size = 1 if split_size==0 else split_size
        mini_batches_special = process_to_pair(mini_batches_special, data.meta_info['rollout'])#.split(split_size)
        mini_batches_log = process_to_pair(mini_batches_log, data.meta_info['rollout'])#.split(split_size)
        mini_batches_special = mini_batches_special.split(split_size)
        mini_batches_log = mini_batches_log.split(split_size)

        metrics = defaultdict(list)

        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches_log = tqdm(mini_batches_log, desc="Train mini-batches", position=1)

            for (mini_batch_log, mini_batch_special) in zip(mini_batches_log, mini_batches_special):
                total_response_tokens = torch.sum(mini_batch_log.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch_log.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches_log, _ = prepare_dynamic_batch(mini_batch_log, max_token_len=max_token_len, mode="pair")
                    micro_batches_special, _ = prepare_dynamic_batch(mini_batch_special, max_token_len=max_token_len, mode="pair")
                else:
                    micro_batches_log = mini_batch_log.split(self.config.micro_batch_size_per_device_for_update)
                    micro_batches_special = mini_batch_special.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches_log = tqdm(micro_batches_log, desc="Update policy", position=2)

                for (micro_batch_log, micro_batch_special) in zip(micro_batches_log, micro_batches_special):
                    # print(f'micro_batches_special size after: [{micro_batch_special.batch}]')
                    # print(f'micro_batches_log size after: [{micro_batch_log.batch}]')
                    
                    micro_batch_log, micro_batch_special = process_pair_to_seq(micro_batch_log), process_pair_to_seq(micro_batch_special)
                    
                    # print(f'micro_batches_special size return: [{micro_batch_special.batch}]')
                    # print(f'micro_batches_log size return: [{micro_batch_log.batch}]')
                    
                    model_inputs_log = {**micro_batch_log.batch, **micro_batch_log.non_tensor_batch}
                    model_inputs_special = {**micro_batch_special.batch, **micro_batch_special.non_tensor_batch}
                    response_mask = model_inputs_log["response_mask"]
                    old_log_probs = model_inputs_log["old_log_probs"]
                    advantages = model_inputs_log["advantages"]
                    # all return: (bsz, response_length)
                    self.special_token_ids = model_inputs_special.get('special_token_ids') or data.meta_info['special_token_ids']
                    special_hidden_state = self._cal_micro_batch_special(model_inputs_special)
                    
                    reward = self.head_module(special_hidden_state) # [batch_size, 1] TODO: 这里需不需要复杂组合
                    log_probs = self._forward_micro_batch(model_inputs_log, temperature=temperature)

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs, 
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    ) 
                    try:
                        bt_loss, bt_metrics = compute_bt_loss(
                            reward, model_inputs_special['ground_truth'],
                        )
                    except Exception as e:
                        bt_loss = torch.zeros_like(pg_loss)
                        bt_metrics = {}
                    
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs_log:
                        ref_log_probs = model_inputs_log["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    bt_loss = self.config.bt_weight * bt_loss

                    loss = loss + bt_loss
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    batch_metrics.update({f"bt/{k}": v for k, v in bt_metrics.items()})
                    batch_metrics["actor/total_loss"] = loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
    
