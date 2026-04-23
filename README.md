# RawHDRV

Official implementation for the paper:

**RawHDRV: HDR Video Reconstruction from Single-Exposure Raw Sequences**

## Overview

RawHDRV reconstructs HDR video frames from single-exposure RAW input sequences.
The repository provides training and testing pipelines based on PyTorch.

## Repository Structure

- `train.py`: training entry
- `test.py`: inference/testing entry
- `models/`: network architecture (`RawHDRV`)
- `data/`: dataset loading and processing utilities
- `config.py`: train/test configuration
- `weight_checkpoints/`: checkpoint directory

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: `requirements.txt` uses CUDA 12.8 nightly PyTorch packages.

## Data Preparation

Prepare dataset directories under `./datasets/` following the paths configured in `config.py`, e.g.:

- `./datasets/Train/`
- `./datasets/Test/`

## Training

```bash
python train.py --model RawHDRV --gpu_id 0
```

## Testing

```bash
python test.py --model RawHDRV --gpu_id 0 --save_image True
```

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{rawhdrv,
	title={RawHDRV: HDR Video Reconstruction from Single-Exposure Raw Sequences}
}
```

## Acknowledgement

This code is based on [RealViformer](https://github.com/Yuehan717/RealViformer.git). Thanks for their excellent work.


