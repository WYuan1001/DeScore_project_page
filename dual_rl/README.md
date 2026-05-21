# Dual RL — Stage 2: Dual-Objective Reinforcement Learning

Stage 2 of DeScore training. Starting from the Stage 1 checkpoint, this stage jointly applies:
- **GRPO loss** to refine CoT reasoning quality (rule-based rollout rewards: format + accuracy + length)
- **Auxiliary BT loss** to calibrate the reward head and prevent "reward drift"

Built on a customized [EasyR1](https://github.com/hiyouga/EasyR1) framework.

---

## File Structure

```
dual_rl/
├── examples/
│   ├── config.yaml                 # Full training configuration
│   ├── train.sh                    # RL training launch script
│   ├── runtime_env.yaml            # Ray runtime env configuration
│   ├── format_prompt/              # Default prompt template
│   │   ├── r1ta.jinja              
│   │   ├── r1v.jinja               
│   │   ├── dapo.jinja              
│   │   └── math.jinja              
│   └── reward_function/
│       └── r1ta_subdim_head.py     # Composite reward function
├── data/                           # Place train.csv and test.csv here
├── verl/                           # Customized RL framework
│   ├── trainer/
│   │   ├── main_head_bt.py         
│   │   ├── ray_trainer_head_bt.py 
│   │   ├── core_algos.py           
│   │   └── config.py               
│   ├── workers/
│   │   ├── fsdp_workers_head.py    
│   │   ├── actor/dp_actor_head.py  
│   │   ├── rollout/vllm_rollout_spmd.py  
│   │   └── reward/function.py      
│   └── utils/
│       ├── dataset.py              # Video + CSV data loading
│       └── checkpoint/             # Checkpoint utilities
└── tests/                          # Unit tests
```

---

## Requirements

```bash
conda env create -f env.yaml
conda activate Descore_rl
pip install -r requirements.txt
```

> `flash_attn` is installed from a local `.whl` file. Build or download the appropriate wheel for your CUDA/PyTorch version before running.

---

## Training

### Step 1: Prepare Checkpoints from Stage 1

Set the Stage 1 checkpoint paths in `examples/train.sh`:

```bash
MODEL_PATH=/path/to/cold_start/output/checkpoint-{step}
HEAD_PATH=/path/to/cold_start/output/checkpoint-{step}/rm_head.pth
```

### Step 2: Prepare Data

Place CSV files in `dual_rl/data/`:

```bash
cp /your/train.csv dual_rl/data/train.csv
cp /your/test.csv  dual_rl/data/test.csv
```

The CSV requires these columns:

| Column | Description |
|---|---|
| `videos` | `[path_A, path_B]` as a parseable Python string |
| `problem` | Text prompt / user instruction |
| `answer` | Pre-collected CoT annotations for each video with ground truth preference|


### Step 3: Launch Training

```bash
cd dual_rl
export VLLM_ATTENTION_BACKEND=FLASHINFER
bash examples/train.sh
```

The launch script runs:

```bash
python3 -m verl.trainer.main_head_bt \
    config=examples/config.yaml \
    data.train_files=./data/train.csv \
    data.val_files=./data/test.csv \
    data.max_prompt_length=6200 \
    data.max_response_length=4096 \
    data.rollout_batch_size=128 \
    data.video_fps=2 \
    algorithm.kl_coef=1.0e-5 \
    worker.actor.global_batch_size=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.head_path=${HEAD_PATH} \
    worker.actor.model.trainable_head=true \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.bt_weight=5e-3 \
    worker.reward.head_ckpt=${HEAD_PATH} \
    worker.rollout.n=8 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=300 \
    trainer.save_checkpoint_path=./results/${EXP_NAME}
```
