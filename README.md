# BAST: Boundary-Aware Self-Training

## Scope

This repository contains the BAST-specific modules and dataset training entry
points for ACDC, LA, and PROMISE12. It is distributed as a compact extension to
the official [BCP](https://github.com/DeepMed-Lab-ECNU/BCP) codebase rather than
a duplicate of the complete BCP repository.

Included files:

- `BAST.py`: ensemble pseudo-labels, boundary weighting, and auxiliary head;
- `ACDC_BAST.py`: ACDC integration;
- `LA_BAST.py`: LA integration;
- `PROMISE12_BAST.py`: PROMISE12 pretraining, self-training, and testing;
- `promise12_dataset.py` and `promise12_metrics.py`: PROMISE12 support.

Dataset files, exact experimental splits, pretrained checkpoints, and logs are
not included.

## Installation

1. Clone the official BCP repository.
2. Copy the six Python files in this repository into the BCP `code/` directory.
3. Install a CUDA-enabled PyTorch/torchvision build for your machine, then run:

```bash
pip install -r requirements.txt
```

The scripts expect the original BCP `dataloaders/`, `networks/`, and `utils/`
directories to remain available under `code/`.

## ACDC

Prepare ACDC using the BCP format, then run from the BCP `code/` directory:

```bash
python ACDC_BAST.py \
  --root_path /path/to/ACDC \
  --labelnum 7 --gpu 0 \
  --ensemble_alpha 0.7 \
  --enable_boundary 1 --lambda_boundary 1.0 \
  --boundary_radius 1 --boundary_rampup 10000 \
  --enable_aux_head 1 --lambda_aux 0.3
```

Use `--labelnum 3` for the 5% setting and `--labelnum 7` for the 10% setting.

## LA

Prepare LA using the BCP format, then run:

```bash
python LA_BAST.py \
  --root_path /path/to/LA \
  --labelnum 8 --gpu 0 \
  --ensemble_alpha 0.7 \
  --enable_boundary 1 --lambda_boundary 1.0 \
  --boundary_radius 1 --boundary_rampup 5000 \
  --enable_aux_head 1 --lambda_aux 0.3
```

Use `--labelnum 4` for the 5% setting and `--labelnum 8` for the 10% setting.

## PROMISE12

PROMISE12 expects processed 2D image/label pairs and user-provided manifests.
Every manifest line contains an image path and its label path relative to the
processed data root:

```text
image/all/Case00/000.npy label/all/Case00/000.npy
```

Required manifest layout:

```text
splits/PROMISE12/
├── train.txt
├── val.txt
├── test.txt
└── 20p/
    ├── labeled.txt
    └── unlabeled.txt
```

Run the stages separately:

```bash
python PROMISE12_BAST.py \
  --data-root /path/to/processed_PROMISE12 \
  --split-root /path/to/splits/PROMISE12 \
  --output-root ./outputs/PROMISE12 \
  --experiment PROMISE12_20p \
  --method bcp --stage pretrain --gpu 0

python PROMISE12_BAST.py \
  --data-root /path/to/processed_PROMISE12 \
  --split-root /path/to/splits/PROMISE12 \
  --output-root ./outputs/PROMISE12 \
  --experiment PROMISE12_20p \
  --method bast --stage selftrain --gpu 0

python PROMISE12_BAST.py \
  --data-root /path/to/processed_PROMISE12 \
  --split-root /path/to/splits/PROMISE12 \
  --output-root ./outputs/PROMISE12 \
  --experiment PROMISE12_20p \
  --method bast --stage test --gpu 0
```

## Citation

```bibtex
@article{Xiao2026BAST,
  author  = {Xiao, Lan and Xi, Qi and Wang, Jing},
  title   = {Boundary-aware self-training for semi-supervised MRI segmentation with bidirectional copy-paste},
  journal = {Biomedical Engineering Letters},
  doi     = {10.1007/s13534-026-00636-2}
}
```

Please also cite the original BCP paper when using these training paths.
