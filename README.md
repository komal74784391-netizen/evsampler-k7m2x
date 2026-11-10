# evsampler

Reference implementation for the paper **"Event-Rate Adaptive Sampling for
Low-Power Depth Sensing"** (3DV 2024).

The repository contains the adaptive sampling policy, the event-rate
controller and the sparse depth completion backbone used in the paper.

## Layout

- `src/sampler/policy.py` - adaptive sampling policy
- `src/sampler/controller.py` - event-rate controller
- `src/models/depthnet.py` - depth completion backbone
- `scripts/train.py`, `scripts/evaluate.py` - training and evaluation
- `configs/base.yaml` - experiment configuration

## Quick start

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/base.yaml
```
