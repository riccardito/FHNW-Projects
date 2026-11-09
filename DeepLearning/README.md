# DeepLearning

**Course:** Deep Learning (FHNW BSc Data Science), plus the code behind my bachelor thesis · **When:** 2021–2022 · **Author:** Riccard Nef

## What this is

Three strands. The two Minichallenges of the Deep Learning module — a CIFAR-10 classifier built up from an MLP, and an image-captioning model on Flickr8k. Then the training pipeline and the explainable-AI evaluation from my 2022 thesis: how stable are Captum heatmaps across seeds, and how well do they match the radiologists' bounding boxes on chest X-rays?

## Contents

| Path | What it does | Result (as written in the notebook) |
|---|---|---|
| `Minichallenge1_del_RiccardNef.ipynb` (+ the `_batchsize64` variant) | CIFAR-10 in PyTorch: MLP → CNN, then learning rate, batch size, L1/L2, dropout, batchnorm, Adam/AdamW; 30 epochs | test accuracy ≈55 % MLP · ≈60 % CNN · ≈68 % with L2 · ≈80 % batchnorm CNN + Adam. Batch size 4 collapsed to 10 % |
| `minichallenge2_riccardnef.ipynb` | "Show and Tell" image captioning: a pretrained CNN encoder (ResNet50 / MobileNet / AlexNet) with a two-layer LSTM decoder, BLEU score; 50 epochs on the CSCS JupyterHub GPU (2.2 min per epoch against 4.7 h on my laptop) | ResNet50 best on BLEU; unseen images captioned poorly. German prose |
| `modeltraining/` | A reusable binary-classifier pipeline for chest X-rays: dataset class, stratified split, AlexNet / ResNet18-50-152 / DenseNet / a hand-written `riciNet`, Adam, wandb logging, a checkpoint saver, flashtorch saliency. Settings in `modeltemplates/params.json` | none stated |
| `xai/Notebooks/XAI/NIH-chest_dataset/` | EDA of the NIH ChestX-ray14 metadata; [Captum](https://captum.ai) heatmaps (Integrated Gradients, GradientSHAP, Occlusion, Noise Tunnel) compared across five to six AlexNet seeds by entropy and distance from the centre of mass. `bbox_eval/` scores heatmap overlap with the NIH bounding boxes (Pneumothorax vs Cardiomegaly) | tables only — about 100 CSV/JSON files of per-model, per-class scores under `results/` and `bbox_eval/results_bb/`. No headline number in prose |
| `xai/Notebooks/XAI/ksbl-wrist_dataset/` | Occlusion heatmaps for five wrist-fracture AlexNets, compared to box labels | none stated |
| `xai/Notebooks/*.ipynb`, `xai/Notebooks/cnn.py` | Scratch: label CSVs, loaders, sample plots | — |

## How to run

Not runnable as is. The Minichallenges download CIFAR-10 through `torchvision` and expect Flickr8k under a local `data_full/`. The pipeline and the XAI notebooks expect NIH ChestX-ray14 under `data/Chest/` and trained `.pth` checkpoints — neither is included. Libraries: `torch`, `torchvision`, `torchxrayvision`, `captum`, `wandb`, `flashtorch`, `seaborn`. No requirements file.

## Data

- **CIFAR-10** — public, via `torchvision`.
- **Flickr8k** — public, from Kaggle; images not included.
- **NIH ChestX-ray14** (`Data_Entry_2017.csv`, `BBox_List_2017.csv`) — public, NIH Clinical Center. Nothing committed except derived score tables, which name NIH's de-identified image files.
- **Wrist X-rays** — a dataset a Swiss hospital provided for the thesis. Not included and not reproducible; the code carries paths only.

## What it shows, and what it doesn't

PyTorch from the ground up, hyper-parameter experiments written down as they happened, transfer learning, HPC use, experiment tracking, and an evaluation design for XAI methods. Against that: scratch notebooks left in, and no write-up of the XAI results in this folder — the thesis text lives elsewhere.

## Status

Minichallenges finished. The thesis code is as it ran in 2022.
