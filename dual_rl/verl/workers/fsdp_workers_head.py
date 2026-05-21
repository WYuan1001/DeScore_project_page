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
The main entry point to run the PPO algorithm
"""

from typing import Literal, Optional, Union, cast
import torch.nn as nn 
import numpy as np
import psutil, os
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights
from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.dataset import process_image, process_video
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import AnyPrecisionAdamW, get_constant_schedule_with_warmup
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from .rollout import vLLMRollout
from .sharding_manager import FSDPVLLMShardingManager
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


class reward_head(nn.Module):  # Fix: Inherit from nn.Module
    def __init__(self, hidden_dim, token_num=1):
        super().__init__()
        self.rm_head = nn.Linear(hidden_dim, token_num, bias=False)
        self.token_num = token_num
    
    @classmethod
    def from_pretrained(cls, hidden_dim, ckpt, token_num=1):
        model = cls(hidden_dim, token_num)
        ckpt_state_dict = torch.load(ckpt, map_location='cpu')
        try:
            ckpt_state_dict['weight'] = ckpt_state_dict.pop('rm_head.weight')
        except:
            ckpt_state_dict['weight'] = ckpt_state_dict.pop('base_model.model.rm_head.weight')
        model.rm_head.load_state_dict(ckpt_state_dict, strict=True)
        return model  

    def forward(self, special_hidden_state):
        if len(special_hidden_state.size()) == 1:
            special_hidden_state = special_hidden_state.unsqueeze(dim=0)  # batch_size, dim
        score = self.rm_head(special_hidden_state) #[b, 3, 3]
        if score.dim() == 3 and score.size(-1) == score.size(-2):
            print('Multi Special Token')
            score = score.diagonal(dim1=1, dim2=2)  # [batch, num_special]
        return score
    
class EmbeddingFreezeCallback:
    def __init__(self, fsdp_module: FSDP, special_token_ids: list[int]):
        self.fsdp_module = fsdp_module
        self.special_token_ids = special_token_ids
        self._snapshot = None  # 保存非目标 token 的 embedding 快照

    def _find_embed_param(self):
        """在 summon_full_params 上下文内调用才能拿到 2D 参数"""
        for name, param in self.fsdp_module.named_parameters():
            if 'embed_tokens.weight' in name:
                return param
        return None

    def _build_freeze_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        mask[self.special_token_ids] = False  # special token 不冻结
        return mask

    def before_step(self):
        if not self.special_token_ids:
            return

        with FSDP.summon_full_params(self.fsdp_module, rank0_only=False):
            param = self._find_embed_param()
            if param is None:
                return
            
            if param.dim() != 2:
                print(f"Warning: embed_tokens shape={param.shape}, still not 2D inside summon_full_params")
                return
            
            freeze_mask = self._build_freeze_mask(param.shape[0], param.device)
            self._snapshot = param.data[freeze_mask].clone()
            self._freeze_mask = freeze_mask

    def after_step(self):
        if not self.special_token_ids or self._snapshot is None:
            return

        with FSDP.summon_full_params(self.fsdp_module, rank0_only=False):
            param = self._find_embed_param()
            if param is None:
                return
            
            with torch.no_grad():
                param.data[self._freeze_mask] = self._snapshot
        
        self._snapshot = None

class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role
        self._cache = {}
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # improve numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self._has_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._has_critic = self.role == "critic"
        self._has_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._has_ref = self.role in ["ref", "actor_rollout_ref"]
        if self._has_actor and self._has_critic:
            raise ValueError("Actor and critic cannot be both initialized.")

        if self.config.actor.disable_kl:
            self._has_ref = False

        self._use_param_offload = False
        self._use_optimizer_offload = False
        self._use_ref_param_offload = False
        if self._has_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_dist_mesh(self.config.actor, "actor")

        if self._has_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
            self._init_dist_mesh(self.config.critic, "critic")

        if self._has_ref:  # NOTE: it seems that manual offload is slower than FSDP offload
            self._use_ref_param_offload = self.config.ref.offload.offload_params

    def _init_dist_mesh(self, config: Union[ActorConfig, CriticConfig], role: Literal["actor", "critic"]):
        world_size = dist.get_world_size()
        # create main device mesh
        fsdp_size = config.fsdp.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        else:  # hsdp
            self.device_mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp")
            )

        # create ulysses device mesh
        if config.ulysses_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // config.ulysses_size, config.ulysses_size),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # validate and normalize config
        if self.config.rollout.n > 1:
            config.global_batch_size *= self.config.rollout.n
            self.print_rank0(f"{role} will use global batch size {config.global_batch_size}.")

        config.global_batch_size_per_device = config.global_batch_size // (world_size // config.ulysses_size)
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")

        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")

        if (
            config.fsdp.enable_cpu_offload
            and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
        ):
            raise ValueError(f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.")
    
    def _setup_partial_embedding_training(self, model, special_token):
        embed_tokens = model.get_input_embeddings()
        
        if isinstance(special_token, str):
            special_token = [special_token]
        
        special_token_ids = []
        for token in special_token:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            
            if token_id == self.tokenizer.unk_token_id and token != self.tokenizer.unk_token:
                self.print_rank0(f"Warning: Token '{token}' not found in vocabulary, skipping.")
                continue
            
            special_token_ids.append(token_id)
        
        if len(special_token_ids) == 0:
            self.print_rank0("No valid special tokens found, skipping partial embedding training.")
            return
        
        # 创建mask
        vocab_size = embed_tokens.weight.shape[0]
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=embed_tokens.weight.device)
        mask[special_token_ids] = True
        
        # 注册hook，只保留特定token的梯度
        def grad_hook(grad):
            if grad is None:
                return None
            masked_grad = grad.clone()
            masked_grad[~mask] = 0  # 其他token的梯度清零
            return masked_grad
        
        # 为input embeddings注册hook
        embed_tokens.weight.register_hook(grad_hook)
        embed_tokens.weight.requires_grad = True
        
        # 处理output embeddings（如果存在且未绑定） 这个部分还是需要再考虑一下
        if hasattr(model, 'get_output_embeddings'):
            output_embed = model.get_output_embeddings()
            if output_embed is not None and hasattr(output_embed, 'weight'):
                # 检查是否与input embeddings共享权重
                if output_embed.weight.data_ptr() != embed_tokens.weight.data_ptr():
                    # 未共享权重，需要单独处理
                    output_embed.weight.register_hook(grad_hook)
                    output_embed.weight.requires_grad = True
                    self.print_rank0("Output embeddings are separate, also set to partial training.")
        
        # 打印调试信息
        token_info = ', '.join([f"'{tok}'(id={tid})" for tok, tid in zip(special_token, special_token_ids)])
        self.print_rank0(f"Partial embedding training enabled for {len(special_token_ids)} tokens: {token_info}")
        
        # 保存信息供后续使用（可选）
        self.trainable_special_token_ids = special_token_ids
        self.trainable_special_tokens = special_token

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool,
        role: Literal["actor", "critic", "ref"],
        tokenizer=None,
        processor=None,
    ) -> None:
        if role != "ref":  # ref model's tokenizer is same as actor
            self.tokenizer = get_tokenizer(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            ) if tokenizer is None else tokenizer
            
            self.processor = get_processor(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            ) if processor is None else processor

            print(f'==== check tokenizer len: [{len(self.tokenizer)}] ====')

            self.model_config = AutoConfig.from_pretrained(
                model_config.model_path,
                trust_remote_code=model_config.trust_remote_code,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **model_config.override_config,
            )

            try:
                self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
            except Exception:
                self.generation_config = GenerationConfig.from_model_config(self.model_config)

            self.print_rank0(f"Model config: {self.model_config}")

        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.bfloat16 # TODO: 不知道为什么这里会出错torch.float32 if role != "ref" else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if role == "critic":
            AutoClass = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys():
            AutoClass = AutoModelForImageTextToText
        else:
            AutoClass = AutoModelForCausalLM
        
        # assert model_config.head_path is not None, print(f'Path of reward_model head is necessary!!!'
        special_token_num = len(self.config.actor.model.add_special_token.split(','))

        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            model = AutoClass.from_pretrained(
                model_config.model_path, 
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )

            try:
                head = reward_head.from_pretrained(4096, model_config.head_path, special_token_num) # 就是model last layer hidden states的维度，这里不知道怎么自动取
            except:
                print(f'Training head from scratch !')
                head = reward_head(4096, special_token_num)
        else:
            with no_init_weights(), init_empty_weights():
                model = AutoClass.from_config(
                    self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=model_config.trust_remote_code,
                )
                head = reward_head(4096, special_token_num)



        model = cast(PreTrainedModel, model)  # lint
        model.tie_weights()  # avoid hanging
        model = model.to(torch_dtype)
        head = head.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        
        trainable_special_tokens = model_config.trainable_special_tokens
        trainable_special_tokens = None if trainable_special_tokens=='None' else trainable_special_tokens

        if role == "ref":
            model.requires_grad_(False)
            head.requires_grad_(False)

        if model_config.freeze_vision_tower:
            if hasattr(model, "model") and hasattr(model.model, "visual"):  # transformers >= 4.52.0
                model.model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            elif hasattr(model, "visual"):  # transformers < 4.52.0
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")
        
        # if trainable_special_tokens is not None:
        fsdp_config.use_orig_params = True
        

        dist.barrier()
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")

        if self.device_mesh.ndim == 2:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP module init")
        fsdp_head_module = FSDP(
            head,
            sharding_strategy=ShardingStrategy.NO_SHARD, #sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=None,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP head module init")

        if role in ["actor", "critic"]:

            trainable_params = [(name, p.shape) for name, p in model.named_parameters() if p.requires_grad]
            self.print_rank0(f"========== Trainable Parameters ({len(trainable_params)}) ==========")
            for name, shape in trainable_params:
                self.print_rank0(f"  {name}: {shape}")
            self.print_rank0(f"  Head trainable: {any(p.requires_grad for p in head.parameters())}")
            self.print_rank0(f"=====================================================")

            self.fsdp_module = fsdp_module
            self.fsdp_head_module = fsdp_head_module

            if trainable_special_tokens is not None and role!='ref':
                # 将token转换为token_ids
                trainable_special_tokens = trainable_special_tokens.split(',')
                special_token_ids = []
                for token in trainable_special_tokens:
                    # 使用convert_tokens_to_ids而不是直接调用tokenizer
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    
                    if token_id == self.tokenizer.unk_token_id and token != self.tokenizer.unk_token:
                        self.print_rank0(f"Warning: Token '{token}' not found in vocabulary, skipping.")
                        continue
            
                    special_token_ids.append(token_id)
                self.trainable_special_token_ids = special_token_ids
                self.embedding_freeze_callback = EmbeddingFreezeCallback(
                    fsdp_module=self.fsdp_module,
                    special_token_ids=self.trainable_special_token_ids,
                )
                self.print_rank0("EmbeddingFreezeCallback registered.")
            else:
                self.embedding_freeze_callback = None

            if model_config.trainable_head:
                param_groups = [
                    {'params': list(filter(lambda p: p.requires_grad, self.fsdp_module.parameters()))},
                    {'params': list(filter(lambda p: p.requires_grad, self.fsdp_head_module.parameters()))},
                ]
            else:
                param_groups = list(filter(lambda p: p.requires_grad, self.fsdp_module.parameters()))

            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    param_groups, # learnable_params, # filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    param_groups, # learnable_params, # filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")

            if optim_config.lr_warmup_steps is not None:
                num_warmup_steps = optim_config.lr_warmup_steps
            else:
                num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)

            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
            print_gpu_memory_usage("After optimizer init")
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")
                if hasattr(self, 'fsdp_head_module'):
                    offload_fsdp_model(self.fsdp_head_module)
                    print_gpu_memory_usage(f"After offload {role}_head model during init")

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")
        else:
            self.ref_fsdp_module = fsdp_module
            self.ref_fsdp_head_module = fsdp_head_module
            if self._use_ref_param_offload:
                offload_fsdp_model(self.ref_fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")
                if hasattr(self, 'ref_fsdp_head_module'):
                    offload_fsdp_model(self.ref_fsdp_head_module)
                    print_gpu_memory_usage(f"After offload {role}_head model during init")

    def _build_rollout(self) -> None:
        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        if self.world_size % tp_size != 0:
            raise ValueError(f"rollout world size {self.world_size} is not divisible by tp size {tp_size}.")

        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
            use_param_offload=self._use_param_offload,
        )
        print_gpu_memory_usage("After vllm init")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, tokenizer=None, processor=None):
        if self._has_critic:
            self._build_model_optimizer(
                model_config=self.config.critic.model,
                fsdp_config=self.config.critic.fsdp,
                optim_config=self.config.critic.optim,
                padding_free=self.config.critic.padding_free,
                role="critic",
                tokenizer=tokenizer,
                processor=processor,
            )

        if self._has_actor:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.actor.fsdp,
                optim_config=self.config.actor.optim,
                padding_free=self.config.actor.padding_free,
                role="actor",
                tokenizer=tokenizer,
                processor=processor,
            )

        if self._has_ref:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.ref.fsdp,
                optim_config=None,
                padding_free=self.config.ref.padding_free,
                role="ref",
                tokenizer=tokenizer,
                processor=processor,
            )

        if self._has_actor:
            if self.config.actor.actor_type == 'single':
                from .actor.dp_actor_head import DataParallelPPOActor  # lazy import

                self.actor = DataParallelPPOActor(
                    config=self.config.actor,
                    actor_module=self.fsdp_module,
                    head_module=self.fsdp_head_module,
                    actor_optimizer=self.optimizer,
                    trainable_head=self.config.actor.model.trainable_head
                )
            else:
                raise Exception(f'The actor_type [{self.config.actor.actor_type}] is not allowed. Please check your configuration.')

        if self._has_critic:
            from .critic.dp_critic import DataParallelPPOCritic  # lazy import

            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._has_rollout:  # must after actor
            self._build_rollout()

        if self._has_ref:
            if self.config.actor.actor_type == 'single':
                from .actor.dp_actor_head import DataParallelPPOActor  # lazy import

                self.ref_policy = DataParallelPPOActor(
                    config=self.config.ref,
                    actor_module=self.ref_fsdp_module,
                    head_module=self.ref_fsdp_head_module,
                )

        if self._has_actor or self._has_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor or self.tokenizer,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
            if hasattr(self, 'fsdp_head_module'):
                load_fsdp_model(self.fsdp_head_module)
        
        if hasattr(self, 'fsdp_head_module') and self.config.actor.model.trainable_head:
            head_group = self.optimizer.param_groups.pop(-1)
            head_states = {}
            for p in self.fsdp_head_module.parameters():
                if p in self.optimizer.state:
                    head_states[p] = self.optimizer.state.pop(p)

        self.checkpoint_manager.save_checkpoint(path, save_model_only)

        if hasattr(self, 'fsdp_head_module') and self.config.actor.model.trainable_head:
            self.optimizer.param_groups.append(head_group)
            self.optimizer.state.update(head_states)
            torch.save(
                {'model': self.fsdp_head_module.state_dict()},
                os.path.join(path, 'rm_head.pth')
            )

        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
            if hasattr(self, 'fsdp_head_module'):
                offload_fsdp_model(self.fsdp_head_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
            if hasattr(self, 'fsdp_head_module'):
                load_fsdp_model(self.fsdp_head_module)
        
        if hasattr(self, 'fsdp_head_module') and self.config.actor.model.trainable_head:
            head_group = self.optimizer.param_groups.pop(-1)

        self.checkpoint_manager.load_checkpoint(path)
        
        if hasattr(self, 'fsdp_head_module') and self.config.actor.model.trainable_head:
            self.optimizer.param_groups.append(head_group)

            head_ckpt_path = os.path.join(path, 'rm_head.pth')
            if os.path.exists(head_ckpt_path):
                ckpt = torch.load(head_ckpt_path, map_location='cpu')
                self.fsdp_head_module.load_state_dict(ckpt['model'])

        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
            if hasattr(self, 'fsdp_head_module'):
                offload_fsdp_model(self.fsdp_head_module)

        if self._use_optimizer_offload:  # avoid OOM in resuming
            offload_fsdp_optimizer(self.optimizer)

    def _process_multi_modal_inputs(self, data: DataProto):
        if "multi_modal_data" not in data.non_tensor_batch:
            return

        try:
            if "uid" in self._cache and not np.all(data.non_tensor_batch["uid"] == self._cache["uid"]):
                self._cache.clear()
        except Exception as e:
            print(f'Check cache error [{e}], because of the special process ..')
            self._cache.clear()

        if "multi_modal_inputs" not in self._cache:
            min_pixels = data.meta_info["min_pixels"]
            max_pixels = data.meta_info["max_pixels"]
            video_fps = data.meta_info["video_fps"]
            batch_multi_modal_inputs = []
            multi_modal_inputs_cache = {}  # avoid repeated processing for n > 1 samples
            for index, multi_modal_data in zip(
                data.non_tensor_batch["uid"], data.non_tensor_batch["multi_modal_data"]
            ):  # process multi modal data per sample
                if index not in multi_modal_inputs_cache:
                    images, videos = [], []
                    video_metadata_lst =[]
                    if "images" in multi_modal_data:
                        for image in multi_modal_data["images"]:
                            images.append(process_image(image, min_pixels, max_pixels))

                    if "videos" in multi_modal_data:
                        for video in multi_modal_data["videos"]:
                            processed_video, meta_info = process_video(video, min_pixels, max_pixels, video_fps, return_fps=True)
                            videos.append(processed_video)
                            # print(f"the video metadata is {meta_info}, the processed video length is {len(processed_video)}")
                            video_metadata_lst.append(meta_info)
                    videos_kwargs = {"video_metadata": video_metadata_lst, "do_sample_frames": False}
                    if len(images) != 0:
                        # it's necessary to add `dict` to properly convert batch features to dict
                        # otherwise the batch features will be converted to dict keys
                        # see https://github.com/hiyouga/EasyR1/pull/339
                        multi_modal_inputs = dict(self.processor.image_processor(images=images,return_tensors="pt"))
                    elif len(videos) != 0:
                        print(f"len of videos is {len(videos[0])} video_fps is {video_fps}")
                        model_inputs = dict(self.processor(
                            videos=videos,text=[""], return_tensors="pt", videos_kwargs=videos_kwargs
                        ))
                        video_input={
                            'video_grid_thw':model_inputs['video_grid_thw'],
                            'pixel_values_videos':model_inputs['pixel_values_videos']
                        }
                        multi_modal_inputs = dict(video_input) 
                    else:
                        multi_modal_inputs = {}
                    multi_modal_inputs = {key: value for key, value in multi_modal_inputs.items() if torch.is_tensor(value)}

                    multi_modal_inputs_cache[index] = multi_modal_inputs

                batch_multi_modal_inputs.append(multi_modal_inputs_cache[index])

            self._cache["uid"] = data.non_tensor_batch["uid"]
            self._cache["multi_modal_inputs"] = np.array(batch_multi_modal_inputs, dtype=object)

        data.non_tensor_batch["multi_modal_inputs"] = self._cache["multi_modal_inputs"]

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        # 上传模型到GPU上
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
            if hasattr(self, 'fsdp_head_module'):
                load_fsdp_model(self.fsdp_head_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data) # bt_loss的计算直接写在这个方法里面了

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            estimated_flops = estimated_flops if not torch.is_tensor(estimated_flops) else estimated_flops.item()
    
            metrics["perf/mfu_actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.lr_scheduler.step()

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )
            # Metrics do not need post processing since their batch size is 1

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
            if hasattr(self, 'fsdp_head_module'):
                offload_fsdp_model(self.fsdp_head_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output
    
    

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_rollout_engine(self):
        self.rollout_sharding_manager.load_vllm_and_sync_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_engine(self):
        self.rollout_sharding_manager.offload_vllm()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._has_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        try:
            add_special_token = prompts.meta_info['add_special_token']
            special_token = prompts.meta_info['special_token']
        except:
            print(f'Normal vllm inference ...')
            add_special_token = False
            special_token = None

        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts, add_special_token=add_special_token, special_token=special_token)
        output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        return output
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_head_reward(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
            if hasattr(self, 'fsdp_head_module'):
                load_fsdp_model(self.fsdp_head_module)

        # we should always recompute old_log_probs when it is HybridEngine
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            trainable = data.meta_info.get('trainable', False)
            output = self.actor.compute_head_reward(data=data, trainable=trainable) # compute_special_hidden_state
            
            output = DataProto.from_dict(
                tensors={"head_reward": output}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
            if hasattr(self, 'fsdp_head_module'):
                offload_fsdp_model(self.fsdp_head_module)

        output = output.to("cpu")
        return output


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data) # compute_special_hidden_state
            
            output = DataProto.from_dict(
                tensors={"old_log_probs": output}, meta_info={"temperature": self.config.rollout.temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_probs(self, data: DataProto):
        assert self._has_ref

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_ref_param_offload:
            load_fsdp_model(self.ref_fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_probs": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_fsdp_module._handle.reshard(True)

        if self._use_ref_param_offload:
            offload_fsdp_model(self.ref_fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_critic"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )
            # Metrics do not need post processing since their batch size is 1

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output
    
