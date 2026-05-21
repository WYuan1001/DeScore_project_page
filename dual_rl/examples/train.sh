#!/bin/bash
# export NCCL_SOCKET_IFNAME=lo # 指定网络接口
export VLLM_ATTENTION_BACKEND=FLASHINFER

set -x

MODEL_PATH=/path/of/base_model
HEAD_PATH=/path/of/head_model
EXP_NAME="test_train"
python3 -m verl.trainer.main_head_bt \
    config=examples/config.yaml \
    data.train_files=./data/train.csv \
    data.val_files=./data/test.csv \
    data.max_prompt_length=6200 \
    data.max_response_length=4096 \
    data.rollout_batch_size=128 \
    data.val_batch_size=16 \
    data.video_fps=2 \
    algorithm.kl_coef=1.0e-5 \
    worker.actor.global_batch_size=8 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.head_path=${HEAD_PATH} \
    worker.actor.model.trainable_head=true\
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.bt_weight=5e-3 \
    worker.reward.head_ckpt=${HEAD_PATH} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.project_name=easyr1_ta \
    worker.rollout.n=8 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.max_num_batched_tokens=14240 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=1 \
    trainer.val_freq=5 \
    trainer.val_before_train=true \
    trainer.save_checkpoint_path=./results/${EXP_NAME} \
    data.min_pixels=3136 \
    data.max_pixels=307200 \
    trainer.total_epochs=300 \
    trainer.find_last_checkpoint=true \
    worker.reward.reward_function=./examples/reward_function/r1ta_subdim_head.py:compute_score \
