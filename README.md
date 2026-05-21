# DeScore

<div align="center">

**Think, then Score: Decoupled Reasoning and Scoring for Video Reward Modeling**

[\[📄 Paper\]](https://arxiv.org/abs/2605.05922) &nbsp; [\[🌐 Project Page\]](https://wyuan1001.github.io/DeScore/)

</div>

---

## Overview

DeScore is a video reward model built on a decoupled *"Think-then-Score"* paradigm:
1. An MLLM (Qwen3-VL-8B) first generates a Chain-of-Thought (CoT) for the input video
2. A learnable `<Reward>` query token + regression head then predicts the final scalar reward **independently** from the CoT generation

Training follows a two-stage framework:
- **Stage 1 — Discriminative Cold Start** (`cold_start/`): LoRA fine-tuning with BT loss on pre-collected CoT data, with random CoT masking for robustness
- **Stage 2 — Dual-Objective RL** (`dual_rl/`): GRPO to refine CoT quality + auxiliary BT loss to calibrate the reward head

---

## Motivation

<p align="center">
  <img src="figure/intro.png" width="85%" alt="DeScore Motivation"/>
</p>

Existing video reward models face a fundamental dilemma:

| Paradigm | Representative Works | Advantage | Disadvantage |
|:---|:---|:---|:---|
| **Discriminative RM** | VideoScore, VideoAlign | Stable optimization via BT/MSE loss | No explicit reasoning; prone to shortcut learning; heavily data-dependent |
| **Generative RM** | UnifiedReward-Thinking, VideoScore2 | CoT improves interpretability & generalization | Training instability; high-variance GRPO gradients; credit assignment difficulty |

As shown in the figure above:
- **(b) Preference Accuracy**: Incorporating CoT enables Generative RMs to outperform Discriminative RMs, highlighting the necessity of explicit thinking for generalization.
- **(c) Training Stability**: Coupling thinking and scoring in one chain forces reliance on GRPO, causing pronounced training fluctuations. BT loss converges smoothly.

DeScore resolves this by **decoupling** reasoning from scoring — the scoring module receives a direct gradient via BT loss, completely bypassing GRPO's high-variance policy gradient. This achieves **+5.4% accuracy on VideoGen-Bench** over the best generative baseline while using **76% less training data**.

---

## Repository Structure

```
DeScore/
├── cold_start/                    # Stage 1: Discriminative Cold Start
│   ├── train_reward.py            # Main training entry
│   ├── trainer_qwen3.py           # Reward model + trainer
│   ├── data.py                    # Data loading & collator
│   ├── utils.py                   # Config dataclasses
│   ├── vision_process.py          # Video preprocessing
│   ├── train.sh                   # Launch script
│   ├── env.yaml                   # Conda environment
│   ├── requirements.txt           # pip dependencies
│   ├── ds_config/                 # DeepSpeed ZeRO configs (zero0/2/3)
│   ├── infer_utils/               # Inference utilities
│   ├── datasets/
│   │   ├── train/                 # Training data (CSV + videos)
│   │   └── eval/                  # Eval benchmark
│   └── model/                     # Place base model here
│
├── dual_rl/                       # Stage 2: Dual-Objective RL
│   ├── inference.py               # Inference entry point
│   ├── inference.sh               # Inference launch script
│   ├── env.yaml                   # Conda environment
│   ├── requirements.txt           # pip dependencies
│   ├── examples/
│   │   ├── config.yaml            # Full training config
│   │   ├── train.sh               # RL training launch script
│   │   ├── format_prompt/         # Jinja2 prompt templates
│   │   └── reward_function/
│   │       └── r1ta_subdim_head.py  # Composite reward function
│   ├── verl/                      # Customized RL framework (Ray + FSDP + vLLM)
│   └── data/                      # Place train/test CSVs here
│
└── figure/                        # Figures for README
```

---

## Quick Start

### 1. Install Dependencies

We recommend using two separate environments for the two stages.

**Stage 1 — Cold Start:**
```bash
conda env create -f cold_start/env.yaml
conda activate Descore_cs
pip install -r cold_start/requirements.txt
```

**Stage 2 — Dual-Objective RL:**
```bash
conda env create -f dual_rl/env.yaml
conda activate qwen3
pip install -r dual_rl/requirements.txt
```

### 2. Download Base Model

```bash
cd cold_start/model
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
cd ../..
```

### 3. Prepare Training Data

Organize your data under `cold_start/datasets/train/`:

```
cold_start/datasets/train/
├── your_data.csv
└── videos/
    ├── video_1_A.mp4
    ├── video_1_B.mp4
    └── ...
```

See [`cold_start/datasets/train/README.md`](cold_start/datasets/train/README.md) and [`cold_start/datasets/train/example.csv`](cold_start/datasets/train/example.csv) for the full CSV schema.

### 4. Stage 1 — Cold Start Training

```bash
cd cold_start
# Edit train.sh: set --model_name_or_path, --meta_data, --output_dir
bash train.sh
```

Output: `checkpoint-*/` (LoRA weights + `rm_head.pth`) under `--output_dir`.

> See [`cold_start/README.md`](cold_start/README.md) for all training arguments.

### 5. Stage 2 — Dual-Objective RL

```bash
cd dual_rl
# Edit examples/train.sh: set MODEL_PATH and HEAD_PATH to the Stage 1 checkpoint
# Place train.csv and test.csv in dual_rl/data/
bash examples/train.sh
```

Output: `results/{EXP_NAME}/global_step_{N}/actor/` (HuggingFace weights + `rm_head.pth`).

> See [`dual_rl/README.md`](dual_rl/README.md) for all training arguments.

### 6. Inference

```bash
cd dual_rl
# Edit inference.sh: set --model_ckpt and --rm_ckpt to the Stage 2 checkpoint
bash inference.sh
```

Or run directly:
```bash
python dual_rl/inference.py \
    --data_path  /path/to/eval.csv \
    --model_ckpt /path/to/checkpoint/actor/huggingface \
    --rm_ckpt    /path/to/checkpoint/actor/rm_head.pth \
    --output     results/output.csv \
    --batch_size 4 \
    --bench_type tabench \
    --special_token "<Reward>"
```

---

## Citation

```bibtex
@article{wang2026think,
  title={Think, then Score: Decoupled Reasoning and Scoring for Video Reward Modeling},
  author={Wang, Yuan and Li, Ouxiang and Xu, Yulong and Liao, Borui and Liang, Jiajun and Li, Jinghan and Wang, Meng and Wang, Xintao and Wang, Pengfei and Liu, Kuien and others},
  journal={arXiv preprint arXiv:2605.05922},
  year={2026}
}
```

## Acknowledgements

- [VideoAlign](https://github.com/KlingAIResearch/VideoAlign): Cold-start training framework
- [EasyR1](https://github.com/hiyouga/EasyR1): Distributed RL training framework
