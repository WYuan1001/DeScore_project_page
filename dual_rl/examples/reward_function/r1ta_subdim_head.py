
### 完整代码 (可以直接复制运行)

import re
import json
from typing import Any, Dict, Optional
import torch.distributed as dist
import torch
import torch.nn as nn



def flatten_reward(rewards): # [rollout, pair] 这样的size的list
    flatten_rewards = []
    for reward in rewards:
        flatten_rewards += reward
    return flatten_rewards

def extract_json_content(text: str) -> Optional[Dict[str, Any]]:
    try:
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        if not match:
            return None
        content = match.group(1).strip()
        code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if code_block_match:
            content = code_block_match.group(1).strip()
        return json.loads(content)
    except (json.JSONDecodeError, AttributeError):
        return None


def format_reward(response: str) -> float:
    data = extract_json_content(response)
    if not data:
        return 0.0
    required_keys = ["dimensional_scores"]
    if all(key in data for key in required_keys):
        return 1.0
    return 0.0

def length_reward(response: str) -> float:
    length  = len(response)
    if length <500:
        return 0.0
    elif length <1000:
        return 0.2
    elif length <1500:
        return 0.4
    elif length <2000:
        return 0.6
    else:
        return 1.0

def cal_acc(response, ground_truth):
    A_pred_score, B_pred_score = response[0], response[1]
    def cal_logit(A, B):
        import math
        reward = 1/(1+math.exp(B-A))
        print(f'Check acc reward! A: [{A}], B: [{B}], Gap: [{A-B}], Reward: [{reward:.4f}]')
        return reward
    if ground_truth == 'A':
        reward = cal_logit(A_pred_score, B_pred_score)
    elif ground_truth == 'B':
        reward = cal_logit(B_pred_score, A_pred_score)
    return reward

def cal_acc_clip(response, ground_truth):
    A_pred_score, B_pred_score = response[0], response[1]
    def cal_logit(A, B):
        import math
        reward = 1/(math.exp(B-A)+1) if A>B else 0.0
        print(f'Check acc reward! A: [{A}], B: [{B}], Gap: [{A-B}], Reward: [{reward:.4f}]')
        return reward
    if ground_truth == 'A':
        reward = cal_logit(A_pred_score, B_pred_score)
    elif ground_truth == 'B':
        reward = cal_logit(B_pred_score, A_pred_score)
    return reward

def sub_dimension_reward(response: str, ground_truth: str, index=0, check=False) -> float:  # 改
    pred_data = extract_json_content(response)
    gt_data = extract_json_content(ground_truth)
    if not pred_data or not gt_data:
        if not pred_data:
            print(f'No pred_data')
        else:
            print(f"No gt_data")
        return 0.0

    total_items = 0
    matched_items = 0
    video_keys = ["video_1_scores", "video_2_scores"][index]
    gt_scores = gt_data.get(video_keys, {})
    pred_scores = pred_data.get('dimensional_scores', {})

    for dim_key, dim_val in gt_scores.items():
        total_items += 1
        if dim_key in pred_scores and pred_scores[dim_key] == dim_val:
            matched_items += 1
    if check:
        print(f'Check sub acc: (1) Pred:[{pred_scores}], (2) GT: [{gt_scores}], (3) match_items: [{matched_items}], (4) Sub_acc: [{matched_items / total_items if total_items > 0 else 0.0}]')

    return matched_items / total_items if total_items > 0 else 0.0


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    reward_inputs = reward_inputs[0]
    for reward_input in reward_inputs: 
        response = reward_input["response"]
        ground_truth = reward_input['ground_truth']

        fmt, sub_acc, len_s = [], [], []
        for idx, resp in enumerate(response):
            if idx==0:
                print(f'========= Check reward ==========')
                check = True
            else:
                check = False
            for i in range(len(resp)):
                if check:
                    print(f'============== check response {i} =================')
                    print(f'Current Response example: [{resp[i]}]')
                fmt += [format_reward(resp[i])]
                sub_acc += [sub_dimension_reward(resp[i], ground_truth[idx][i], index=i, check=check)]
                len_s += [length_reward(resp[i])]
                if check:
                    print(f'(1) Format: [{fmt[-1]}], (2) Sub_Acc: [{sub_acc[-1]}], (3) Length_s: [{len_s[-1]}], (4) Overall: [{0.7*sub_acc[-1]+0.1*fmt[-1]+0.2*len_s[-1]}]')


        overall = [0.7*per_acc+0.1*per_fmt+0.2*per_len for per_acc, per_fmt, per_len in zip(sub_acc, fmt, len_s)]
        scores += [{
            "overall": round(per_overall, 3),
            "format": round(per_fmt, 3),
            "accuracy": round(per_acc, 3),
            "length": round(per_lens, 3)
        } for per_overall, per_acc, per_fmt, per_lens in zip(overall, sub_acc, fmt, len_s)]
    return scores