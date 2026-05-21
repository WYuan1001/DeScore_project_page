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

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer
import numpy as np
from collections import defaultdict
import torch.distributed as dist

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]
    

class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionHeadRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        video_path = [[[list(item[n][i].values())[0] for i in range(2)] for n in range(len(item))] for item in data.non_tensor_batch['multi_modal_data']] # 128*rollout*2
        for i in range(len(data)): 
            response_str = []
            cur_response_length = []
            for n in range(len(response_ids[0])):
                cur_response_length.append([response_length[i][n][j].item() for j in range(len(response_length[i][n]))])  # avoid tensor indexing error
                valid_response_ids = [response_ids[i][n][j][:cur_response_length[n][j]] for j in range(len(response_ids[i][n]))]
                response_str.append([self.tokenizer.decode(
                    valid, skip_special_tokens=self.config.skip_special_tokens
                ) for valid in valid_response_ids])
            if 'special_hidden_state' in data.batch.keys():
                reward_inputs.append(
                    {
                        'special_hidden_state': data.batch['special_hidden_state'][i],
                        "response": response_str,
                        "response_length": cur_response_length,
                        "ground_truth": data.non_tensor_batch["ground_truth"][i],
                        'image': video_path[i]
                    }
                )
            elif 'head_reward' in data.batch.keys():
                reward_inputs.append(
                    {
                        'head_reward': data.batch['head_reward'][i],
                        "response": response_str,
                        "response_length": cur_response_length,
                        "ground_truth": data.non_tensor_batch["ground_truth"][i],
                        'image': video_path[i]
                    }
                )
            else:
                reward_inputs.append(
                    {
                        "response": response_str,
                        "response_length": cur_response_length,
                        "ground_truth": data.non_tensor_batch["ground_truth"][i],
                        'image': video_path[i]
                    }
                )
            
        scores = self.reward_fn([reward_inputs, self.config.head_ckpt]) # 算reward 输出应该是128*8*2 这样的reward，最好还是resize成这些都拉平
        flatten_cur_response_length = []
        for reward in reward_inputs:
            temp = reward['response_length']
            for per in temp:
                flatten_cur_response_length += per

        reward_tensor = torch.zeros_like(data.batch["responses"].view((-1,)+data.batch['responses'].size()[3:]), dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(flatten_cur_response_length[i])  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)
        return reward_tensor, reward_metrics

class PairFunctionHeadRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        video_path = [[[list(item[n][i].values())[0] for i in range(2)] for n in range(len(item))] for item in data.non_tensor_batch['multi_modal_data']] # 128*rollout*2
        for i in range(len(data)): 
            response_str = []
            cur_response_length = []
            for n in range(len(response_ids[0])): 
                cur_response_length.append([response_length[i][n][j].item() for j in range(len(response_length[i][n]))])  # avoid tensor indexing error
                valid_response_ids = [response_ids[i][n][j][:cur_response_length[n][j]] for j in range(len(response_ids[i][n]))]
                response_str.append([self.tokenizer.decode(
                    valid, skip_special_tokens=self.config.skip_special_tokens
                ) for valid in valid_response_ids])
            reward_inputs.append(
                {
                    'special_hidden_state': data.batch['special_hidden_state'][i],
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    'image': video_path[i]
                }
            )
        scores, orders = self.reward_fn([reward_inputs, self.config.head_ckpt]) 
        flatten_cur_response_length = []
        for reward, order in zip(reward_inputs, orders):
            temp = reward['response_length'] 
            new_temp = [[temp[i.item()][0], temp[j.item()][1]] for i, j in zip(order[:, 0], order[:, 1])]
            for per in new_temp:
                flatten_cur_response_length += per

        reward_tensor = torch.zeros_like(data.batch["responses"].view((-1,)+data.batch['responses'].size()[3:]), dtype=torch.float32) # TODO: 这个应该就是没有pair那个纬度了吧，确认一下
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(flatten_cur_response_length[i])  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)
        reward_metrics['order'] = orders
        return reward_tensor, reward_metrics

class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        video_path = [[[list(item[n][i].values())[0] for i in range(2)] for n in range(len(item))] for item in data.non_tensor_batch['multi_modal_data']] # 128*rollout*2

        for i in range(len(data)): 
            response_str = []
            cur_response_length = []
            for n in range(len(response_ids[0])): #
                cur_response_length.append([response_length[i][n][j].item() for j in range(len(response_length[i][n]))])  # avoid tensor indexing error
                valid_response_ids = [response_ids[i][n][j][:cur_response_length[n][j]] for j in range(len(response_ids[i][n]))]
                response_str.append([self.tokenizer.decode(
                    valid, skip_special_tokens=self.config.skip_special_tokens
                ) for valid in valid_response_ids])
            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    'image': video_path[i]
                }
            )
            
        scores = self.reward_fn(reward_inputs) 
        flatten_cur_response_length = []
        for reward in reward_inputs:
            temp = reward['response_length']
            for per in temp:
                flatten_cur_response_length += per

        reward_tensor = torch.zeros_like(data.batch["responses"].view((-1,)+data.batch['responses'].size()[3:]), dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(flatten_cur_response_length[i])  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)
        return reward_tensor, reward_metrics

