# Local WSL2 Setup for the DGX Spark Hugging Face Fine-Tune Notebook

This documents the exact setup used on this machine to run the Colab notebook locally:

https://colab.research.google.com/drive/1hV6Gcz8vBRS9t0bYkBp6W1ne_yqG6mJx?usp=sharing

The downloaded notebook filename is:

```text
NVIDIA-DGX-Spark-hugging_face_llm_full_fine_tune_tutorial-VIDEO.ipynb
```

This setup is intentionally specific to this environment:

- Windows machine running WSL2
- Ubuntu inside WSL2
- NVIDIA GeForce RTX 5070
- Windows NVIDIA driver exposing CUDA into WSL
- Python environment managed with `uv`

It is not intended to be a generic Linux or multi-GPU setup.

## What Was Checked First

The machine is WSL2:

```bash
uname -a
cat /proc/version
```

The GPU driver files already existed in the WSL integration directory:

```bash
ls -l /usr/lib/wsl/lib/nvidia-smi /usr/lib/wsl/lib/libcuda.so*
```

On WSL2, `nvidia-smi` usually comes from the Windows NVIDIA driver, not from installing normal Ubuntu NVIDIA driver packages. On this machine it was already present here:

```text
/usr/lib/wsl/lib/nvidia-smi
```

It was not on `PATH`, so I added a symlink:

```bash
sudo ln -sf /usr/lib/wsl/lib/nvidia-smi /usr/local/bin/nvidia-smi
```

Then I made the WSL NVIDIA libraries discoverable by the dynamic linker:

```bash
printf '/usr/lib/wsl/lib\n' | sudo tee /etc/ld.so.conf.d/wsl-nvidia.conf >/dev/null
sudo ldconfig
```

Verification:

```bash
nvidia-smi
ldconfig -p | rg 'libcuda|libnvidia-ml'
```

The detected GPU was:

```text
NVIDIA GeForce RTX 5070
Driver Version: 581.57
CUDA Version: 13.0
VRAM: 12227 MiB
Compute capability: 12.0
```

## Workspace Created

The notebook workspace was created at:

```text
/home/gnu/colab-local-dgx-spark
```

The notebook was downloaded directly from Google Drive:

```bash
mkdir -p /home/gnu/colab-local-dgx-spark
curl -L 'https://drive.google.com/uc?export=download&id=1hV6Gcz8vBRS9t0bYkBp6W1ne_yqG6mJx' \
  -o /home/gnu/colab-local-dgx-spark/NVIDIA-DGX-Spark-hugging_face_llm_full_fine_tune_tutorial-VIDEO.ipynb
```

## Python Environment

The system Python was Python 3.14, which is too new for many ML wheels. I used `uv` to create a Python 3.12 virtual environment:

```bash
cd /home/gnu/colab-local-dgx-spark
uv venv --python 3.12 .venv
source .venv/bin/activate
```

CUDA PyTorch was installed from the CUDA 12.8 PyTorch wheel index:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

The notebook dependencies were installed:

```bash
uv pip install \
  transformers \
  trl \
  datasets \
  accelerate \
  gradio \
  huggingface_hub \
  matplotlib \
  tqdm \
  ipywidgets \
  jupyterlab \
  notebook \
  spaces
```

The verified environment included:

```text
torch 2.11.0+cu128
torch CUDA runtime 12.8
transformers 5.11.0
trl 1.5.1
datasets 5.0.0
accelerate 1.13.0
gradio 6.17.3
huggingface_hub 1.18.0
```

## GPU Verification

PyTorch CUDA verification:

```bash
cd /home/gnu/colab-local-dgx-spark
source .venv/bin/activate
python scripts/check_gpu.py
```

Expected result on this machine:

```text
torch: 2.11.0+cu128
torch CUDA runtime: 12.8
CUDA available: True
GPU: NVIDIA GeForce RTX 5070
Compute capability: (12, 0)
CUDA matmul OK
```

## Jupyter

JupyterLab was started inside the notebook workspace:

```bash
cd /home/gnu/colab-local-dgx-spark
./scripts/start_jupyter.sh
```

The launcher binds to all interfaces by default:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

This makes it reachable through Tailscale on this machine:

```text
http://100.92.158.40:8888
```

The Jupyter token is printed when the server starts. To list running servers and tokens:

```bash
cd /home/gnu/colab-local-dgx-spark
source .venv/bin/activate
jupyter server list
```

To force local-only binding instead:

```bash
JUPYTER_IP=127.0.0.1 ./scripts/start_jupyter.sh
```

It serves notebooks from:

```text
/home/gnu/colab-local-dgx-spark
```

Then open the printed `http://127.0.0.1:8888/lab?...` URL in the browser and open:

```text
NVIDIA-DGX-Spark-hugging_face_llm_full_fine_tune_tutorial-VIDEO.ipynb
```

## Hugging Face Requirement

The notebook uses this gated model:

```text
google/gemma-3-270m-it
```

The dataset downloads without authentication, but the model does not. Before the model cells run, accept the model terms here:

https://huggingface.co/google/gemma-3-270m-it

Then log in locally:

```bash
cd /home/gnu/colab-local-dgx-spark
source .venv/bin/activate
hf auth login
hf auth whoami
```

If `HF_TOKEN` is already available in the shell, the setup script can log in automatically with:

```bash
HF_TOKEN=hf_your_token ./SETUP.sh
```

The token still needs access to the gated Gemma model.

## VRAM Note

The notebook says it wants roughly 16 GB of GPU memory. This RTX 5070 exposes about 12 GB VRAM in WSL2.

If fine-tuning runs out of memory, change the notebook `SFTConfig` batch settings from:

```python
per_device_train_batch_size=16
per_device_eval_batch_size=16
gradient_checkpointing=False
```

to:

```python
per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=4
gradient_checkpointing=True
```

This lowers peak VRAM while keeping the effective training batch close to the notebook default.

## One-Shot Setup

From this repo:

```bash
cd /home/gnu/llm-fine-tune
./SETUP.sh
```

To also start Jupyter after setup:

```bash
START_JUPYTER=1 ./SETUP.sh
```

To log into Hugging Face during setup:

```bash
HF_TOKEN=hf_your_token ./SETUP.sh
```

To rebuild the Python environment from scratch:

```bash
RESET_ENV=1 ./SETUP.sh
```
