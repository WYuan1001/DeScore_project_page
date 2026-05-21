# Cold Start — Stage 1: Discriminative Cold Start

Stage 1 of DeScore training. Fine-tunes Qwen3-VL-8B with LoRA using Bradley-Terry (BT) loss on pre-collected CoT preference data. A `<Reward>` query token is appended to the input; its hidden state is projected by a regression head into a scalar reward.

---

## File Structure

```
cold_start/
├── train_reward.py          # Main training entry point
├── trainer_qwen3.py         # Qwen3VLRewardModelBT + VideoVLMRewardTrainer
├── data.py                  # DataConfig, dataset loading, data collator
├── utils.py                 # TrainingConfig, ModelConfig, PEFTLoraConfig dataclasses
├── vision_process.py        # Video frame sampling and pixel processing
├── train.sh                 # Training launch script
├── ds_config/
│   ├── zero0.json           # ZeRO Stage 0
│   ├── zero2.json           # ZeRO Stage 2 (optimizer sharding)
│   └── zero3.json           # ZeRO Stage 3 (full param sharding)
├── infer_utils/
│   ├── qwen2_vl.py          
│   ├── qwen3_vl.py          
│   └── torch_functional.py  
├── datasets/
│   ├── train/
│   │   ├── example.csv      # Example training CSV
│   │   ├── videos/          # Paired video files
│   │   └── README.md        # Data format details
│   └── eval/
│       └── README.md       
└── model/                   # Place Qwen3-VL-8B-Instruct here
```

---

## Data Format 
The CSV must contain the following columns:
`videos`, `problem`, `durations`, `prompt`, `CoT_A`, `CoT_B`, `GSB`
| Column | Type | Description |
|---|---|---|
| `videos` | str | Path to all videos within the pair (e.g., `["./videos/video_1_A.mp4", "./videos/video_1_B.mp4"]`) |
| `problem` | str | User instruction for the video |
| `durations` | str | Duration of the video |
| `prompt` | str | Text description for the video |
| `CoT_A` | str | Chain-of-Thought for video A |
| `CoT_B` | str | Chain-of-Thought for video B |
| `GSB` | str | Good/Same/Bad label: `A`, `B`, `same`, or `invalid` |

See [`datasets/train/example.csv`](datasets/train/example.csv) for a concrete example.

---

## Training

### Step 1: Install Dependencies

```bash
conda env create -f env.yaml
conda activate Descore_cs
pip install -r requirements.txt
```

### Step 2: Download Base Model

```bash
mkdir -p model && cd model
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
cd ..
```

### Step 3: Configure `train.sh`

Edit `train.sh` to update the key paths:

```bash
--model_name_or_path ./model/Qwen3-VL-8B-Instruct   # base model path
--meta_data          ./datasets/train/your_data.csv  # training CSV
--meta_data_test     ./datasets/train/your_data.csv  # validation CSV
--output_dir         ./output                        # checkpoint output dir
```

### Step 4: Launch

```bash
cd cold_start
bash train.sh
```

The default `train.sh`:

```bash
deepspeed --master_port=28500 train_reward.py \
    --use_special_tokens True \
    --lora_enable True \
    --lora_r 64 --lora_alpha 128 \
    --freeze_vision_tower False \
    --tune_merger True \
    --fps 2 \
    --max_frame_pixels 200704 \
    --model_name_or_path ./model/Qwen3-VL-8B-Instruct \
    --meta_data "./datasets/train/example.csv" \
    --output_dir test_result \
    --eval_dim "GSB" \
    --loss_type "bt" \
    --learning_rate 2e-6 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --deepspeed ds_config/zero0.json \
    --enable_drop True
```

---

## Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| **Paths** | | |
| `--model_name_or_path` | `./model/Qwen3-VL-8B-Instruct` | Base model path |
| `--meta_data` | `./datasets/train/example.csv` | Training CSV |
| `--meta_data_test` | same | Validation CSV |
| `--output_dir` | `test_result` | Output directory |
| **LoRA** | | |
| `--lora_enable` | `True` | Enable LoRA |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `128` | LoRA alpha |
| `--vision_lora` | `False` | Apply LoRA to vision tower |
| `--lora_namespan_exclude` | `['lm_head', 'rm_head', 'embed_tokens']` | Modules excluded from LoRA |
| **Freezing** | | |
| `--freeze_vision_tower` | `False` | Freeze visual encoder |
| `--freeze_llm` | `False` | Freeze LLM backbone |
| `--tune_merger` | `True` | Tune vision-LLM merger module |
| **Video** | | |
| `--fps` | `2` | Video sampling FPS |
| `--max_frame_pixels` | `200704` | Max pixels per frame (448×448) |
| `--sample_type` | `"uniform"` | Frame sampling strategy |
| **Reward Head** | | |
| `--use_special_tokens` | `True` | Add `<Reward>` token for scoring |
| `--reward_token` | `"special"` | Token used for reward: `last`, `mean`, `special` |
| `--output_dim` | `1` | Reward head output dimension |
| `--eval_dim` | `"GSB"` | Dimension to evaluate: `GSB` |
| **Loss** | | |
| `--loss_type` | `"bt"` | Loss type: `bt`, `btt`, `margin`, etc. |
| `--enable_drop` | `True` | Enable random CoT masking (recommended: `True`) |
| **Optimization** | | |
| `--bf16` | `True` | bfloat16 training |
| `--learning_rate` | `2e-6` | Base learning rate |
| `--merger_lr` | `2e-6` | Merger module learning rate |
| `--vision_lr` | `2e-6` | Vision tower learning rate |
| `--special_token_lr` | `2e-6` | `<Reward>` token embedding learning rate |
| `--num_train_epochs` | `2` | Training epochs |
| `--per_device_train_batch_size` | `1` | Batch size per GPU |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation |
| `--max_length` | `6144` | Max sequence length |
| `--gradient_checkpointing` | `True` | Enable gradient checkpointing |
| **Saving** | | |
| `--deepspeed` | `ds_config/zero0.json` | DeepSpeed config (`zero0`, `zero2`, `zero3`) |
| `--save_only_model` | `True` | Save model weights only |
| `--seperate_save_model` | `True` | Save LoRA adapter and `rm_head.pth` separately |
| `--save_full_model` | `False` | Save full merged model |
| `--save_epochs` | `0.5` | Save frequency (epochs) |
| `--eval_epochs` | `0.1` | Eval frequency (epochs) |

---

## DeepSpeed Config Selection

| File | ZeRO Stage | Notes |
|---|---|---|
| `ds_config/zero0.json` | Stage 0 | No sharding; highest GPU memory usage |
| `ds_config/zero2.json` | Stage 2 | Optimizer + gradient sharding |
| `ds_config/zero3.json` | Stage 3 | Full parameter sharding; lowest GPU memory |

Recommended: `zero0.json` or `zero2.json` for 8B model on 8× A100 80GB.

---

## Notes

- `--enable_drop True` is critical: it randomly masks the CoT during training, preventing the scoring module from ignoring raw video features.
