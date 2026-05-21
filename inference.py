import os
import re
import copy
import json
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import pandas as pd
import random
import ast
import numpy as np
import traceback
from verl.models.transformers.qwen3_vl import get_rope_index
import verl.utils.torch_functional as VF
from torch import nn
from verl.utils.dataset import process_video
import torch.distributed as dist
TEMP = '''
**Role**: You are a professional Video-Text Alignment Judge. Your job is to strictly evaluate whether a generated video matches the user's text prompt.

**Inputs**:
- **Text Prompt**: [insert prompt here]
- **Video**: <video>

---

### Step 1: Thinking Process (Chain of Thought)
*You must perform your reasoning in plain text BEFORE generating the final `<answer>` block. Follow these three stages of thought:*

**1. Prompt Decoding & Visual Expectation**
Analyze the text prompt to establish a ""Gold Standard"" for evaluation before looking at the video.
*   **Explicit Elements**: List specific entities, actions, texts, or objects mentioned.
*   **Implicit/Abstract Translation**: If the prompt contains abstract concepts (e.g., ""loneliness"", ""cinematic"", ""chaos""), translate them into concrete visual indicators (e.g., ""dark lighting"", ""shallow depth of field"", ""fast camera movement"").

**2. Visual Evidence Extraction**
Look at the video and describe what you actually see objectively, without forcing a match to the prompt.
*   What/Who is the main subject?
*   What are the exact actions and movements?
*   Are there any visual artifacts, hallucinations, or typos in generated text?

**3. Dimensional Analysis & Score Calculation**
Compare the Visual Evidence against the Prompt Expectations across 5 dimensions.
*   **Subject (Primary)**: Accuracy of main entities and text generation (spelling).
*   **Dynamics (Primary)**: Accuracy of actions, physics, and temporal flow.
*   **Camera (Secondary)**: Movement, angles, shot types.
*   **Environment (Secondary)**: Background, setting, lighting.
*   **Style (Secondary)**: Art style, aesthetic, visual quality.

*Sub-Dimension Scoring:* Assign **2** (Perfect), **1** (Partial/Minor flaws), **0** (Failure/Hallucination), or **N/A** (Not applicable).

*Overall Score Interval Logic (1.00 - 5.00):*
*   **[5.00] Excellent**: Subject & Dynamics = 2. All Secondary = 2 or N/A. No flaws.
*   **[4.00 - 4.99] Good**: Subject & Dynamics = 2. Flaws only in Secondary dimensions.
*   **[3.00 - 3.99] Fair**: One Primary is 1, OR core intent is met but with noticeable secondary flaws.
*   **[2.00 - 2.99] Poor**: One Primary is 0. Major hallucinations present.
*   **[1.00 - 1.99] Failed**: Both Primaries are 0. Complete mismatch.

---

### Step 2: Final Output
After completing your thinking process, output the final scores strictly in the following JSON format enclosed within `<answer>` tags. Do not include any reasoning inside the JSON.

<answer>
```json
{
  ""dimensional_scores"": {
    ""subject"": ""0, 1, 2, or N/A"",
    ""dynamics"": ""0, 1, 2, or N/A"",
    ""camera"": ""0, 1, 2, or N/A"",
    ""environment"": ""0, 1, 2, or N/A"",
    ""style"": ""0, 1, 2, or N/A""
  },
  ""overall_score"": 1.00-5.00
}
```
</answer>
    '''

class reward_head(nn.Module):  # Fix: Inherit from nn.Module
    def __init__(self, hidden_dim, ckpt):
        super().__init__()
        self.rm_head = nn.Linear(hidden_dim, 1, bias=False)
        ckpt_state_dict = torch.load(ckpt, map_location='cpu')
        if 'model' in ckpt_state_dict:
            ckpt_state_dict = ckpt_state_dict['model']
        try:
            ckpt_state_dict['weight'] = ckpt_state_dict.pop('rm_head.weight')
        except:
            print('Try different key name ...')
            ckpt_state_dict['weight'] = ckpt_state_dict.pop('base_model.model.rm_head.weight')

        self.rm_head.load_state_dict(ckpt_state_dict, strict=True)  

    @torch.no_grad()
    def forward(self, special_hidden_state):
        if len(special_hidden_state.size()) == 1:
            special_hidden_state = special_hidden_state.unsqueeze(dim=0)  # batch_size, dim
        score = self.rm_head.to(special_hidden_state.device)(special_hidden_state).squeeze(0) 
        return score

def convert_pair_to_single(df_pair_anno, bench_type):
    if bench_type == 'videoalign':
        df_A = df_pair_anno[['path_A', 'A_model', 'prompt', 'fps_A', 'num_frames_A']]
        df_A.columns = ['path', 'model', 'prompt', 'fps', 'num_frames']

        df_B = df_pair_anno[['path_B', 'B_model', 'prompt', 'fps_B', 'num_frames_B']]
        df_B.columns = ['path', 'model', 'prompt', 'fps', 'num_frames']

        df_single = pd.concat([df_A, df_B], axis=0)
        df_single = df_single.drop_duplicates(subset=['path'])
        df_single = df_single.sort_values(by=['path'])

        df_single = df_single.reset_index(drop=True)
    elif bench_type == 'genai':
        # prompt,left_model,right_model,vote_type,left_video_path,right_video_path
        df_A = df_pair_anno[['left_video_path', 'left_model', 'prompt']]
        df_A.columns = ['path', 'model', 'prompt']
        df_B = df_pair_anno[['right_video_path', 'right_model', 'prompt']]
        df_B.columns = ['path', 'model', 'prompt']
        df_single = pd.concat([df_A, df_B], axis=0)
        df_single = df_single.drop_duplicates(subset=['path'])
        df_single = df_single.sort_values(by=['path'])
        df_single = df_single.reset_index(drop=True)
    elif bench_type == 'react':
        # Unnamed: 0,path_A,path_B,A_model,B_model,prompt,GSB_MQ,GSB_MQ_A_reasoning,GSB_MQ_B_reasoning
        df_A = df_pair_anno[['path_A', 'A_model', 'prompt']]
        df_A.columns = ['path', 'model', 'prompt']
        
        df_B = df_pair_anno[['path_B', 'B_model', 'prompt']]
        df_B.columns = ['path', 'model', 'prompt']
        df_single = pd.concat([df_A, df_B], axis=0)
        df_single = df_single.drop_duplicates(subset=['path'])
        df_single = df_single.sort_values(by=['path'])
        df_single = df_single.reset_index(drop=True)
    elif bench_type == 'videofeedback':
        # video_path,prompt,visual_score,t2v_score,phy_score,thinking,video_fps,frame_count,duration
        df_single = df_pair_anno[['video_path', 'prompt', 'video_fps']]
        df_single.columns = ['path', 'prompt', 'fps']
        
        df_single = df_single.drop_duplicates(subset=['path'])
        df_single = df_single.sort_values(by=['path'])
        df_single = df_single.reset_index(drop=True)
    elif bench_type == 'tabench':
        # video_path,prompt,visual_score,t2v_score,phy_score,thinking,video_fps,frame_count,duration
        df_A = df_pair_anno[['video_A', 'prompt', 'answer']]
        df_A.columns = ['path', 'prompt', 'answer']

        df_B = df_pair_anno[['video_B', 'prompt', 'answer']]
        df_B.columns = ['path', 'prompt', 'answer']

        df_single = pd.concat([df_A, df_B], axis=0)
        df_single = df_single.drop_duplicates(subset=['path'])
        df_single = df_single.sort_values(by=['path'])

        df_single = df_single.reset_index(drop=True)
    
    return df_single

def convert_single_to_pair(df_pair_anno, df_single_pred, bench_type):
    score_dict = {}
    keys_to_store = ["Reward", 'CoT']

    for i, row in df_single_pred.iterrows():
        score_dict[row["path"]] = {k: row[k] for k in keys_to_store}

    for key in keys_to_store:
        df_pair_anno[f"{key}_A"] = 0.0
        df_pair_anno[f"{key}_B"] = 0.0

    for i, row in df_pair_anno.iterrows():
        for key in keys_to_store:
            df_pair_anno.at[i, f"{key}_A"] = score_dict[row["video_A"]][key]
            df_pair_anno.at[i, f"{key}_B"] = score_dict[row["video_B"]][key]
    return df_pair_anno


def seed_everything(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class PromptImageDataset(Dataset):
    def __init__(self, data, processor, mllm_name):
        self.data = data
        self.processor = processor
        self.mllm_name = mllm_name
        self.video_path =  self.data['path'].to_list()
        if 'problem' in self.data:
            self.prompt_list = self.data['problem'].to_list()
        else:
            self.prompt_list = [TEMP.replace('[insert prompt here]', prompt) for prompt in self.data['prompt'].to_list()]

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        from qwen_vl_utils import process_vision_info
        
        PATH = self.video_path[idx]
        PROMPT = self.prompt_list[idx]
        
        results = []
        
        
        try:
            if not os.path.exists(PATH):
                raise FileNotFoundError(f"Video not found: {PATH}")
            
            if os.path.getsize(PATH) == 0:
                raise ValueError(f"Video file is empty: {PATH}")
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT.split('<video>')[0]},
                    {"type": "video", "video": PATH},
                    {"type": "text", "text": PROMPT.split('<video>')[1]},
                ],
            }]

            _, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16, # self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True
            )
            
            if video_inputs is None or len(video_inputs) == 0:
                raise ValueError(f"Video processing failed: empty video_inputs for {PATH}")
            
            mm_data = {"video": video_inputs}
            message = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
                
            results.append({
                "status": "success",
                "request_id": f'{idx}',
                "input": {
                    "prompt": message,
                    "multi_modal_data": mm_data,
                    'mm_processor_kwargs': video_kwargs
                },
                'raw_prompt': PROMPT,
                "video_path": PATH,
            })
            
        except Exception as e:
            error_trace = traceback.format_exc()
            results.append({
                "status": "failed",
                "request_id": f'{idx}',
                "error": str(e),
                "error_trace": error_trace,
                "PATH": PATH,
            })
    
        return results

def collate_fn(batch):
    all_requests = []
    for sample_requests in batch:
        all_requests.extend(sample_requests)
    
    successful = [r for r in all_requests if r["status"] == "success"]
    failed = [r for r in all_requests if r["status"] == "failed"]
    
    return {
        "successful": successful,
        "failed": failed,
    }

def process_data_after_generation(processor, requests, results, pad_token_id, device='cpu'):
    from qwen_vl_utils import process_vision_info
    
    batch_size = len(requests)
    
    all_input_ids = []
    all_attention_mask = []
    all_position_ids = []
    all_mm_inputs = []
    
    for i, request in enumerate(requests):
        video_path = request['video_path']
        prompt_text = request['raw_prompt']
        response_text = results[request['request_id']]
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text.split('<video>')[0]},
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt_text.split('<video>')[1]},
                {"type": "text", "text": f'{response_text}{args.special_token}'}, 
            ],
        }]

        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        if i==0:
            print(f'Check text prompt: [{text_prompt}]')
        
        processed_video, meta_info = process_video(
                video_path, 3136, 200704, 2, return_fps=True
            )
        videos_kwargs = {"video_metadata": [meta_info], "do_sample_frames": False}
        
        model_inputs = processor(
            videos=[processed_video],
            text=[text_prompt],
            add_special_tokens=False,
            return_tensors="pt",
            videos_kwargs=videos_kwargs
        )
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        
        vision_position_ids = get_rope_index(
            processor,
            input_ids=input_ids,
            video_grid_thw=model_inputs.get("video_grid_thw"),
            attention_mask=attention_mask,
        ) 
        text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
        position_ids = torch.cat([text_position_ids, vision_position_ids], dim=0)
        
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_position_ids.append(position_ids)
        
        mm_input = {}
        for key in ["pixel_values_videos", "video_grid_thw"]:
            if key in model_inputs:
                mm_input[key] = model_inputs[key]
        all_mm_inputs.append(mm_input)
    
    
    max_seq_len = max(ids.size(0) for ids in all_input_ids)
    batch_input_ids = torch.full(
        (batch_size, max_seq_len),
        pad_token_id,
        dtype=torch.long,
        device=device
    )
    batch_attention_mask = torch.zeros(
        (batch_size, max_seq_len),
        dtype=torch.long,
        device=device
    )
    
    batch_position_ids = torch.zeros(
        (batch_size, 4, max_seq_len),
        dtype=torch.long,
        device=device
    )
    
    for i in range(batch_size):
        seq_len = all_input_ids[i].size(0)
        batch_input_ids[i, :seq_len] = all_input_ids[i]
        batch_attention_mask[i, :seq_len] = all_attention_mask[i]
        batch_position_ids[i, :, :seq_len] = all_position_ids[i]
    
    if batch_position_ids.dim() == 3:  # qwen2vl mrope
            batch_position_ids = batch_position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)
    
    merged_mm_inputs = {}
    if all_mm_inputs and "pixel_values_videos" in all_mm_inputs[0]:
        merged_mm_inputs["pixel_values_videos"] = torch.cat(
            [mm["pixel_values_videos"] for mm in all_mm_inputs], dim=0
        ).to(device)
        merged_mm_inputs["video_grid_thw"] = torch.cat(
            [mm["video_grid_thw"] for mm in all_mm_inputs], dim=0
        ).to(device)
    return {'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask,
        'position_ids': batch_position_ids,
        'mm_inputs': merged_mm_inputs}

def extract_hidden_states_for_special_token(
    model, 
    input_ids, 
    attention_mask, 
    position_ids,
    special_token_id,
    multi_modal_data=None
):
    batch_size, seq_length = input_ids.shape

    model_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'use_cache': False,
        }
    
    if multi_modal_data is not None:
        model_kwargs.update(multi_modal_data)

    with torch.no_grad():
        outputs = model.model(**model_kwargs)
    last_hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]

    special_token_mask = (input_ids == special_token_id)  # (batch_size, seq_length)
    
    pooled_hidden_states_list = []
    for i in range(batch_size):
        sample_hidden = last_hidden_states[i]  # (seq_length, hidden_dim)
        sample_mask = special_token_mask[i]     # (seq_length,)
        sample_special_hidden = sample_hidden[sample_mask.to(sample_hidden.device)]  # (num_special_in_sample, hidden_dim)
        
        if sample_special_hidden.size(0) == 0:
            print(f"Warning: Sample {i} has no special token, using zero vector")
            pooled_hidden_states_list.append(torch.zeros(last_hidden_states.size(-1), device=last_hidden_states.device))
        else:
            pooled_hidden_states_list.append(sample_special_hidden.mean(dim=0))
    
    pooled_hidden_states = torch.stack(pooled_hidden_states_list, dim=0)  # (batch_size, hidden_dim)
    our_rm_head = reward_head(pooled_hidden_states.size()[-1], args.rm_ckpt).to(last_hidden_states.device, dtype=pooled_hidden_states.dtype)
    reward = our_rm_head(pooled_hidden_states)
    return reward

def run_inference_and_extract_hidden_states(
    llm,
    hf_model,
    processor,
    requests,
    sampling_params,
    max_tokens,
    extract_fn,
    pad_token_id,
    special_token_id
):
    if not requests:
        return {}, {}
    
    sampling_params.max_tokens = max_tokens
    request_ids = [r['request_id'] for r in requests]
    inputs = [r['input'] for r in requests]
    
    llm_outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
    
    in_toks = [len(o.prompt_token_ids) for o in llm_outputs]
    out_toks = [len(o.outputs[0].token_ids) for o in llm_outputs]
    print(f"  → Token count: avg {sum(in_toks)/len(in_toks):.0f}+{sum(out_toks)/len(out_toks):.0f}, "
            f"max {max(in_toks)} + {max(out_toks)}, total {sum(in_toks):,} + {sum(out_toks):,}")
    
    results = {}
    for i, output in enumerate(llm_outputs):
        text = extract_fn(output.outputs[0].text)
        results[request_ids[i]] = text
    
    processed_data = process_data_after_generation(processor, requests, results, pad_token_id, device=hf_model.device)

    reward = extract_hidden_states_for_special_token(
        model=hf_model,
        input_ids=processed_data['input_ids'],
        attention_mask=processed_data['attention_mask'],
        position_ids=processed_data['position_ids'],
        multi_modal_data=processed_data['mm_inputs'],
        special_token_id=special_token_id   
    )
    result_rewards = {}
    for i, r in enumerate(reward):
        result_rewards[request_ids[i]] = r.item()
    return results, result_rewards

def start_evaluation_qwen(args, mllm_path, batch_size=1, max_tokens=4096, save_every=50):
    from qwen_vl_utils import process_vision_info

    llm_config = dict(
        model=mllm_path,
        max_model_len=25600,
        max_num_seqs=batch_size * 2,  
        gpu_memory_utilization=0.7,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={"video": 1},
        mm_encoder_tp_mode="data",
    )

    if "Qwen2_5" in args.mllm:
        pass
    elif "Qwen3" in args.mllm:
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        llm_config.update(
            dtype="bfloat16",
            enable_expert_parallel='_A' in args.mllm,
            distributed_executor_backend="mp",
        )
    
    llm = LLM(**llm_config)
    processor = AutoProcessor.from_pretrained(mllm_path, trust_remote_code=True)
    
    print("Loading HuggingFace model for hidden state extraction...")
    from transformers import Qwen3VLForConditionalGeneration
    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        mllm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    hf_model.eval()
    
    special_token_id = processor.tokenizer.encode(args.special_token, add_special_tokens=False)[0]
    pad_token_id = processor.tokenizer.pad_token_id
    print(f"Special token: '{args.special_token}' (id={special_token_id})")
    print(f"PAD token id: {pad_token_id}")
    
    sampling_params = SamplingParams(
        temperature=0.0,
        repetition_penalty=1.05,
        stop_token_ids=[],
    )

    data = pd.read_csv(args.data_path)
    data = convert_pair_to_single(data, bench_type=args.bench_type)
    dataset = PromptImageDataset(data, processor, args.mllm)
    
    print(f"Total samples: {len(dataset)} (will generate {len(dataset) * 2} requests)")

    temp_output_path = args.output_path.replace('.csv', '/temp')
    # hidden_states_path = args.output_path.replace('.csv', '_hidden_states.pt')
    
    cache_data = []
    
    error_log_path = args.output_path.replace('.csv', '_errors.log')

    cached_results = {}
    cached_reward_results = {}
    cached_hidden_states = {}
    failed_requests = {}

    def save_checkpoint(name):
        if failed_requests:
            with open(error_log_path, 'a') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Failed Requests: {len(failed_requests)}\n")
                f.write("=" * 80 + "\n\n")
                for req_id, error_info in failed_requests.items():
                    f.write(f"Request ID: {req_id}\n")
                    f.write(f"Video Path: {error_info.get('video_path', 'N/A')}\n")
                    f.write(f"Error: {error_info.get('error', 'N/A')}\n")
                    f.write(f"Trace:\n{error_info.get('error_trace', 'N/A')}\n")
                    f.write("-" * 80 + "\n\n")
        
        RESULT, REWARD = [], []
        valid_indices = []

        for idx in range(len(dataset)):
            result = cached_results.get(f'{idx}', '')
            reward = cached_reward_results.get(f'{idx}', '')
            RESULT.append(result)
            REWARD.append(reward)
            
            if result != '':
                valid_indices.append(idx)

        temp_data = data.loc[valid_indices].copy()
        temp_data['CoT'] = [RESULT[i] for i in valid_indices]
        temp_data['Reward'] = [REWARD[i] for i in valid_indices]
        os.makedirs(temp_output_path, exist_ok=True)
        temp_data.to_csv(os.path.join(temp_output_path, f'{name}.csv'), index=False)
        
        
    def extract_fn(text):
        if 'thinking' in args.mllm.lower():
            return text.split('</think>\n\n')[-1] if '</think>\n\n' in text else ""
        return text

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        # num_workers=8,
        # prefetch_factor=2
    )

    completed_since_save = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        successful_requests = batch["successful"]
        failed_batch = batch["failed"]
        
        for req in failed_batch:
            req_id = req["request_id"]
            failed_requests[req_id] = {
                "video_path": req.get("video_path"),
                "error": req.get("error"),
                "error_trace": req.get("error_trace"),
            }
            print(f"\n[ERROR] {req_id} preprocessing failed: {req.get('error')}")
        
        pending_requests = [
            r for r in successful_requests 
            if r["video_path"] not in cache_data and r["request_id"] not in failed_requests
        ]
        
        if not pending_requests:
            continue
        
        try:
            results, reward_results = run_inference_and_extract_hidden_states(
                llm=llm,
                hf_model=hf_model,
                processor=processor,
                requests=pending_requests,
                sampling_params=sampling_params,
                max_tokens=max_tokens,
                extract_fn=extract_fn,
                pad_token_id=pad_token_id,
                special_token_id=special_token_id
            )
            
            cached_results.update(results)
            cached_reward_results.update(reward_results)
            completed_since_save += len(pending_requests)
            
        except Exception as e:
            print(f"\n[ERROR] Batch {batch_idx} failed: {e}")
            traceback.print_exc()
            for req in pending_requests:
                failed_requests[req['request_id']] = {
                    "video_path": req.get("video_path"),
                    "error": str(e),
                    "error_trace": traceback.format_exc(),
                }
            continue
        
        if completed_since_save >= save_every:
            save_checkpoint(f'batch_{batch_idx}')
            completed_since_save = 0
            cached_results = {}
    
    print("\n[Final] Saving results...")
    
    if cached_results or cached_hidden_states:
        save_checkpoint('final')
    
    all_data = pd.concat([pd.read_csv(os.path.join(temp_output_path, file_name)) 
                          for file_name in os.listdir(temp_output_path) if file_name.endswith('.csv')])
    all_data = convert_single_to_pair(data, all_data)
    all_data.to_csv(args.output_path, index=False)
    print(f"Saved final CSV to {args.output_path}")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm', type=str, default='Qwen3_VL_8B_Instruct', help="""
        Qwen-series: Qwen2_5_VL_72B, Qwen3_VL_8B_Instruct, Qwen3_VL_8B_Thinking, Qwen3_VL_32B_Instruct, Qwen3_VL_32B_Thinking, Qwen3_VL_30B_A3B_Instruct, Qwen3_VL_30B_A3B_Thinking, Qwen3_VL_235B_A22B_Instruct, Qwen3_VL_235B_A22B_Thinking
        Gemini-series: Gemini_2_5_Flash
    """)
    parser.add_argument('--data_path', type=str, help="path of the csv file")
    parser.add_argument('--model_ckpt', type=str, default=None, help='model checkpoint')
    parser.add_argument('--rm_ckpt', type=str, help="checkpoint of the reward model")
    parser.add_argument('--special_token', type=str, default="<Reward>", help="Special token to add after generation")
    parser.add_argument('--output_path', type=str, default="results.csv")
    parser.add_argument('--bench_type', type=str, default="tabench")
    parser.add_argument('--batch_size', type=int, default=4, help="Number of samples per batch (will generate 2x requests)")
    parser.add_argument('--save_every', type=int, default=100, help="Save checkpoint every N requests")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device("cuda")
    seed_everything(args.seed)

    MLLMs = {
        "Qwen3_VL_8B_Instruct"        : args.model_ckpt,
    }
    
    if "Qwen" in args.mllm: 
        start_evaluation_qwen(args, MLLMs[args.mllm], batch_size=args.batch_size, save_every=args.save_every)
    
   