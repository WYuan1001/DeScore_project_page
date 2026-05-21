# DeScore

> **Think, then Score: Decoupled Reasoning and Scoring for Video Reward Modeling**
> [\[Paper\]](https://arxiv.org/abs/2605.05922) [\[Project Page\]](https://wyuan1001.github.io/DeScore/)

DeScore is a video reward model that uses a decoupled *"Think-then-Score"* paradigm: an MLLM (Qwen3-VL-8B) first generates a Chain-of-Thought (CoT), then a learnable `<Reward>` token + regression head outputs the final scalar reward. Training follows two stages:
1. **Stage 1 — Discriminative Cold Start** (`cold_start/`): LoRA fine-tuning with BT loss on pre-collected CoT data
2. **Stage 2 — Dual-Objective RL** (`dual_rl/`): GRPO for CoT quality + auxiliary BT loss for reward calibration

---

## Motivation

Existing video reward models face a fundamental dilemma:
![image](figure/intro.pdf)
First, (b) Preference Accuracy shows that incorporating CoT enables Generative RMs to outperform Discriminative RMs, highlighting the necessity of explicit thinking for generalization. Second, (c) Training Stability reveals that coupling thinking and scoring requires the final score to be optimized through GRPO loss, leading to pronounced training fluctuations. In contrast, discriminative training with BT loss exhibits smooth convergence. Motivated by these findings, DeScore introduces a decoupled "think-then-score" paradigm that effectively leverages the generalization benefits of CoT reasoning while preserving the training stability inherent to discriminative optimization.

DeScore resolves this dilemma with a **decoupled "Think-then-Score"** design:
- The MLLM backbone generates an explicit CoT, providing fine-grained semantic rationales
- A dedicated `<Reward>` query token + regression head predicts the scalar reward **independently** from the CoT
- The scoring module is optimized directly via stable BT loss, completely **bypassing** GRPO's high-variance policy gradient

This decoupling preserves the generalization benefits of CoT reasoning while maintaining the optimization stability of discriminative regression — achieving **+5.4% accuracy on VideoGen-Bench** over the best generative baseline while using **76% less training data**.

---

## Repository Structure

```
DeScore/
├── inference.py                    # Inference entry point
├── inference.sh                    # Inference launch script
├── cold_start/                # Stage 1: Cold start training
│   ├── train_reward.py        # Main training entry
│   ├── trainer_qwen3.py       # Reward model + trainer
│   ├── data.py                # Data loading & collator
│   ├── utils.py               # Config dataclasses
│   ├── vision_process.py      # Video preprocessing
│   ├── train.sh               # Launch script
│   ├── ds_config/             # DeepSpeed ZeRO configs 
│   ├── infer_utils/           
│   ├── datasets/
│   │   ├── train/             # Training data (CSV + videos)
│   │   └── eval/              # Eval benchmark
│   └── model/                 # Place base model here
│
└── dual_rl/                   # Stage 2: RL fine-tuning
    ├── examples/
    │   ├── config.yaml        # Full training config
    │   ├── train.sh           # RL training launch script
    │   ├── format_prompt/     # Jinja2 prompt templates
    │   └── reward_function/
    │       └── r1ta_subdim_head.py  # Composite reward function
    ├── verl/                  # Customized RL framework 
    └── data/                  # Place train/test CSVs here
```

---

## Quick Start

### 1. Install Dependencies

We recommend using two separate environments for the two stages.

**Stage 1 (cold start):**
```bash
conda env create -f cold_start/env.yaml
conda activate Descore_cs
pip install -r cold_start/requirements.txt
```

**Stage 2 (dual RL):**
```bash
conda env create -f dual_rl/env.yaml
conda activate Descore_rl
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
The CSV requires columns: `videos`, `problem`, `durations`, `prompt`, `CoT_A`, `CoT_B`, `GSB`.
See [`cold_start/datasets/train/README.md`](cold_start/datasets/train/README.md) and [`cold_start/datasets/train/example.csv`](cold_start/datasets/train/example.csv) for details.

### 4. Stage 1 — Cold Start Training

```bash
cd cold_start
# Edit train.sh to set --model_name_or_path, --meta_data, --output_dir
bash train.sh
```

See [`cold_start/README.md`](cold_start/README.md) for all training arguments.

### 5. Stage 2 — Dual-Objective RL

```bash
cd dual_rl
# Edit examples/train.sh: set MODEL_PATH and HEAD_PATH to the Stage 1 checkpoint
# Place train.csv and test.csv in dual_rl/data/
bash examples/train.sh
```
See [`dual_rl/README.md`](dual_rl/README.md) for all training arguments.

### 6. Inference

```bash
cd dual_rl
# Edit inference.sh: set --model_ckpt and --rm_ckpt to the Stage 2 checkpoint
bash inference.sh
```

Or run directly:
```bash
python inference.py \
    --data_path /path/to/eval.csv \
    --model_ckpt /path/to/checkpoint/actor/huggingface \
    --rm_ckpt /path/to/checkpoint/actor/rm_head.pth \
    --output results/output.csv \
    --batch_size 4 \
    --bench_type tabench \
    --special_token "<Reward>"
```

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
