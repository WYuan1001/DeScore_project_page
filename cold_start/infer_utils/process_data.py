from qwen_vl_utils.vision_process import fetch_video
import os
import torch

def process_video(
    video: str, min_pixels, max_pixels, video_fps, return_fps, image_patch_size=16
):
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps} #,"video_fps": video_fps
    # print(f"the video is {video}")
    processed_video, meta_info = fetch_video(vision_info, return_video_metadata=return_fps, image_patch_size=image_patch_size)
    
    if return_fps:
        return processed_video, meta_info
    else:
        return processed_video