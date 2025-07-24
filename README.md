# Quick Start

## 1. Download Models
Download the following models:
- `showlab/showo512x512`
- `microsoft/phi-1_5`
- `showlab/magvitv2`

## 2. Modify Configuration Files
Update the paths in these configuration files:
- `configs/showo_demo_512x512.yaml`: Paths for all three models
- `showo512x512/config.json`: Path for the phi-1.5 model

> **Note**: The ShowO model must be downloaded as it requires vocabulary modifications.

## 3. Install Dependencies
Install required packages from `requirements.txt`.

## 4. Prepare Data and UniCToken Second-Stage Results
Refer to the [UniCToken] documentation for details.

---

# Parameter Description

This example uses the `adrien_brody` dataset. You need to modify:
- Dataset path in `train_grpo.py`
- Second-stage results save path
- `num_gen`: Controls group size
- `batch_size`: Default is 1
- `epoch`: Controls training iterations

Also modify `save_dir` in `grpo.py` - this is the temporary output directory for generated images.

---

# Code Description

- `train_grpo.py`: Handles parameter initialization, model setup, and ShowO architecture modifications.
- `grpo.py`: Implements the `unic_grpo` class with three reward functions:
  1. Self evaluation (`reward1`)
  2. CLIP evaluation (`reward2`)
  3. GPT-4o evaluation (`reward3`)

---

# Start Training

- Standard training:
  ```bash
  python train_grpo.py
  ```

- DeepSpeed training (4 GPUs):
  ```bash
  deepspeed train_grpo.py --num_gpus 4
  ```

---
