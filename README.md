# DLC data managenet

## Installation

1. Install DeepLabCut first, following the official instructions for your
   platform/GPU: https://deeplabcut.github.io/DeepLabCut/docs/installation.html
   (this pulls in torch, torchvision, and the right CUDA build for your machine —
   pinning it inside this package would conflict with that).

   On this project's dev machine: `deeplabcut==3.0.1`, `torch==2.13.0`,
   `torchvision==0.28.0`, CUDA 13.x via `nvidia-cudnn-cu13`/`nvidia-cublas`.
   A full snapshot of that environment is in `environment-lock.yml`
   (`conda env create -f environment-lock.yml`) if you want an exact match
   rather than a fresh DLC install.

2. Then install dlc_manager itself:

   pip install --pre git+https://github.com/you/dlc_manager.git

