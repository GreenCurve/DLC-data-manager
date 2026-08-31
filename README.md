# DLC data manager

## Quickstart

**1. Install DeepLabCut** (GPU/platform-specific, do this first):

```bash
pip install 'deeplabcut[gui]'   # local machine — needed for labeling
pip install deeplabcut          # Colab / headless — training & inference only
```

**2. Install dlc_manager:**

```bash
pip install --pre git+https://github.com/GreenCurve/DLC-data-manager.git
```

(`--pre` is required — pins a pydantic pre-release.)

**3. (Optional) W&B logging:**

```bash
wandb login
```

In Colab, use a Secrets-panel API key instead — see Colab notes below.

---

Exact dev-machine environment (`deeplabcut==3.0.1`, `torch==2.13.0`, CUDA 13.x):
`conda env create -f environment-lock.yml`