import pdb
from dataclasses import dataclass
from typing import Optional, List, Union

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
# from vision_process import process_vision_info
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from utils import save_video
import random 

system_prompt = '''

'''

@dataclass
class DataConfig:
    meta_data: str = "/path/to/dataset/meta_data.csv"
    meta_data_test: str = None
    max_frame_pixels: int = 240 * 320
    num_frames: float = None
    fps: float = 2.0
    p_shuffle_frames: float = 0.0
    p_color_jitter: float = 0.0
    eval_dim: str = "VQ"
    prompt_template_type: str = "none"
    add_noise: bool = False
    sample_type: str = "uniform"
    use_cot: bool = True
    enable_drop: bool = False

def convert_GSB_csv_to_reward_data(example, eval_dims="GSB", max_pixels=448 * 448, fps=2.0, 
                                   num_frames=None, sample_type="uniform"):
    """
    Convert Good/Same/Bad csv data to reward data.

    Args:
        example (dict): A dataframe containing the GSB csv data.
        data_dir (str): The directory path to the video files.
        eval_dim (str): The dimension to evaluate ("VQ"/"MQ"/"TA").
        max_pixels (int): The maximum number of pixels allowed for videos.
        num_frames (float): Number of frames.
        prompt_template_type (str): The type of prompt template to use ("none"/"simple"/"video_score").

    Returns:
        dict: A dictionary containing the reward data.
    """
    import ast
    import torch.distributed as dist
    video_path = ast.literal_eval(example['videos'])
    A_data = {'video': video_path[0], 
                'max_pixels': max_pixels,
                'fps': fps if num_frames is None else None,
                'sample_type': sample_type,
                'prompt': example['problem'], 
                'cot': example["CoT_A"],
                'special_token': '<Reward>'}
    B_data = {'video': video_path[1], 
                'max_pixels': max_pixels,
                'fps': fps if num_frames is None else None,
                'sample_type': sample_type,
                'prompt': example['problem'], 
                'cot': example["CoT_B"],
                'special_token': '<Reward>'}

    chosen_labels = []
    A_scores = []
    B_scores = []
    import torch.distributed as dist
    
    ### chosen_label: 1 if A is chosen, -1 if B is chosen, 0 if tied.
    ### 22 if invalid. ooaaeeaa o.O
    try:
        if example[f"{eval_dims}"] is not None:
            if example[f"{eval_dims}"] == "A":
                chosen_label = 1
            elif example[f"{eval_dims}"] == "B":
                chosen_label = -1
            elif example[f"{eval_dims}"] == "invalid":
                chosen_label = 22
            else:
                chosen_label = 22
        else:
            chosen_label = 22
    except Exception as e:
        chosen_label = 22

    chosen_labels.append(chosen_label)
    if f"MOS_A_{eval_dims}" in example and f"MOS_B_{eval_dims}" in example:
        try:
            A_score = example[f"MOS_A_{eval_dims}"] if example[f"MOS_A_{eval_dims}"] is not None else 0.0
            B_score = example[f"MOS_B_{eval_dims}"] if example[f"MOS_B_{eval_dims}"] is not None else 0.0
        except Exception as e:
            A_score = 0.0
            B_score = 0.0
        A_scores.append(A_score)
        B_scores.append(B_score)
    else:
        A_scores.append(0.0)
        B_scores.append(0.0)

    chosen_labels = torch.tensor(chosen_labels, dtype=torch.long)
    A_scores = torch.tensor(A_scores, dtype=torch.float)
    B_scores = torch.tensor(B_scores, dtype=torch.float)
    metainfo_idx = None
    if 'metainfo_idx' in example:
        metainfo_idx = example['metainfo_idx']

    return {"A_data": A_data, "B_data": B_data, 
            "A_scores": A_scores, "B_scores": B_scores, 
            "chosen_label": chosen_labels,
            "metainfo_idx": metainfo_idx,}

class QWen3VLDataCollator():
    def __init__(self, processor, add_noise=False, p_shuffle_frames=0.0, p_color_jitter=0.0, enable_drop=False):
        self.processor = processor
        self.add_noise = add_noise
        self.set_noise_step = None

        self.p_shuffle_frames = p_shuffle_frames
        self.p_color_jitter = p_color_jitter
        self.enable_drop = enable_drop

        self.noise_adder = None

    def _clean_message(self, message, random_num=None):
        """
        remove unnecessary keys from message(very very necessary)
        """
        if not self.enable_drop:
            out_message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video", 
                                "video": message["video"], 
                                "max_pixels": message["max_pixels"], 
                                "fps": message["fps"],
                                "sample_type": "uniform",
                            },
                            {"type": "text", "text": f'{message["prompt"]}{message["cot"]}{message["special_token"]}'},
                        ],
                    }
                ]
        else:
            if random_num<0.5:
                print(f'current data mask cot')
                out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video", 
                            "video": message["video"], 
                            "max_pixels": message["max_pixels"], 
                            "fps": message["fps"],
                            "sample_type": message['sample_type'],
                        },
                        {"type": "text", "text": f'{message["prompt"]}{message["special_token"]}'},
                    ],
                }]
            else:
                print(f'current data none mask')
                out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video", 
                            "video": message["video"], 
                            "max_pixels": message["max_pixels"], 
                            "fps": message["fps"],
                            "sample_type": message['sample_type'],
                        },
                        {"type": "text", "text": f'{message["prompt"]}{message["cot"]}{message["special_token"]}'},
                    ],
                }]


        if out_message[0]["content"][0]["fps"] is None:
            out_message[0]["content"][0].pop("fps")
        
        return out_message


    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded

    def __call__(self, features, enable_noise=True):
        """
        Preprocess inputs to token sequences and return a batch
        """
        # try:
        features_A = []
        features_B = []
        # check if we have a margin. If we do, we need to batch it as well
        # has_margin = "margin" in features[0]
        has_idx = "metainfo_idx" in features[0] and features[0]["metainfo_idx"] is not None

        for idx, feature in enumerate(features):
            random_num = random.random() if self.enable_drop else None   
            features_A.append(self._clean_message(feature["A_data"], random_drop=self.enable_drop, random_num=random_num))
            features_B.append(self._clean_message(feature["B_data"], random_drop=self.enable_drop, random_num=random_num))

        # import pdb; pdb.set_trace()
        image_inputs_A, video_inputs_A, video_kwargs_A = process_vision_info(features_A, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        image_inputs_B, video_inputs_B, video_kwargs_B = process_vision_info(features_B, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if video_inputs_A is not None:
            video_inputs_A, video_metadatas_A = zip(*video_inputs_A)
            video_inputs_A, video_metadatas_A = list(video_inputs_A), list(video_metadatas_A)
        else:
            video_metadatas_A = None

        if video_inputs_B is not None:
            video_inputs_B, video_metadatas_B = zip(*video_inputs_B)
            video_inputs_B, video_metadatas_B = list(video_inputs_B), list(video_metadatas_B)
        else:
            video_metadatas_B = None
        
        batch_A = self.processor(
            text=self.processor.apply_chat_template(features_A, tokenize=False, add_generation_prompt=True),
            images=image_inputs_A,
            videos=video_inputs_A,
            padding=True,
            return_tensors="pt",
            do_resize=False,
            video_metadata=video_metadatas_A,
            **video_kwargs_A
        )
        batch_B = self.processor(
            text=self.processor.apply_chat_template(features_B, tokenize=False, add_generation_prompt=True),
            images=image_inputs_B,
            videos=video_inputs_B,
            padding=True,
            return_tensors="pt",
            do_resize=False,
            video_metadata=video_metadatas_B,
            **video_kwargs_B,
        )

        # pdb.set_trace()
        def find_subsequence(input_ids, subsequence):
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            
            for idx in range(len(input_ids) - len(subsequence) + 1):
                if input_ids[idx:idx + len(subsequence)] == subsequence:
                    return idx+3 
            return -1  

        indexs_A, indexs_B = [], []
        for input_ids_A, input_ids_B in zip(batch_A['input_ids'], batch_B['input_ids']):
            input_ids_A = batch_A["input_ids"].squeeze(0)  
            input_ids_B = batch_B["input_ids"].squeeze(0) 
            subsequence = [522, 9217, 397] # find response mask

            # Find the subsequence in input_ids_A and input_ids_B
            indexs_A.append(find_subsequence(input_ids_A, subsequence))
            indexs_B.append(find_subsequence(input_ids_B, subsequence))


        len_A, len_B = batch_A["input_ids"].shape[1], batch_B["input_ids"].shape[1]
        max_len = max(batch_A["input_ids"].shape[1], batch_B["input_ids"].shape[1])
        batch_A["input_ids"], batch_A["attention_mask"] = self._pad_sequence(batch_A["input_ids"], batch_A["attention_mask"], max_len, "right")
        batch_B["input_ids"], batch_B["attention_mask"] = self._pad_sequence(batch_B["input_ids"], batch_B["attention_mask"], max_len, "right")
        batch_A_response_mask = torch.zeros_like(batch_A['input_ids']).to(batch_A['input_ids'].dtype)
        batch_B_response_mask = torch.zeros_like(batch_B['input_ids']).to(batch_A['input_ids'].dtype)
        for i, (index_A, index_B) in enumerate(zip(indexs_A, indexs_B)):
            batch_A_response_mask[i, index_A: len_A] = 1
            batch_B_response_mask[i, index_B: len_B] = 1
        
        chosen_label = torch.stack([torch.tensor(feature["chosen_label"]) for feature in features])

        A_scores = torch.stack([torch.tensor(feature["A_scores"]) for feature in features])
        B_scores = torch.stack([torch.tensor(feature["B_scores"]) for feature in features])
        
        batch = {
            "A": batch_A,
            "B": batch_B,
            "return_loss": True,
            "chosen_label": chosen_label,
            "A_scores": A_scores,
            "B_scores": B_scores,
            'A_response_mask': batch_A_response_mask,
            'B_response_mask': batch_B_response_mask
        }

        if has_idx:
            metainfo_idx = torch.stack([torch.tensor(feature["metainfo_idx"]) for feature in features])
            batch["metainfo_idx"] = metainfo_idx

        return batch

        # except Exception as e:
        #     print(f"Error processing batch: {e} in reading.")
        #     # get next batch
        #     return None
