#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/colab-local-dgx-spark}"
NOTEBOOK_ID="1hV6Gcz8vBRS9t0bYkBp6W1ne_yqG6mJx"
NOTEBOOK_FILE="NVIDIA-DGX-Spark-hugging_face_llm_full_fine_tune_tutorial-VIDEO.ipynb"
NOTEBOOK_URL="https://drive.google.com/uc?export=download&id=${NOTEBOOK_ID}"
EXPECTED_GPU="NVIDIA GeForce RTX 5070"

log() {
  printf '\n[setup] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '[setup] Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

log "Checking required base commands"
require_cmd sudo
require_cmd curl
require_cmd rg

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

require_cmd uv

log "Configuring NVIDIA WSL2 command and libraries"
if [[ ! -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  printf '[setup] /usr/lib/wsl/lib/nvidia-smi is missing. Update/install the Windows NVIDIA driver with WSL CUDA support first.\n' >&2
  exit 1
fi

sudo ln -sf /usr/lib/wsl/lib/nvidia-smi /usr/local/bin/nvidia-smi
printf '/usr/lib/wsl/lib\n' | sudo tee /etc/ld.so.conf.d/wsl-nvidia.conf >/dev/null
sudo ldconfig

log "Checking GPU"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "$GPU_NAME" != "$EXPECTED_GPU" ]]; then
  printf '[setup] Expected "%s" but found "%s". This setup script is intentionally machine-specific.\n' "$EXPECTED_GPU" "$GPU_NAME" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader

log "Creating notebook workspace at $WORKSPACE"
mkdir -p "$WORKSPACE/scripts"
cd "$WORKSPACE"

log "Downloading notebook"
curl -L "$NOTEBOOK_URL" -o "$NOTEBOOK_FILE"

log "Patching notebook for current TRL/Transformers integration settings"
python - <<'PY'
import json
from pathlib import Path

notebook_path = Path("NVIDIA-DGX-Spark-hugging_face_llm_full_fine_tune_tutorial-VIDEO.ipynb")
notebook = json.loads(notebook_path.read_text())
changed = False

for cell in notebook.get("cells", []):
    source = cell.get("source")
    if not isinstance(source, list):
        continue

    updated_source = []
    for line in source:
        replacements = {
            "per_device_train_batch_size=16": "per_device_train_batch_size=1",
            "per_device_eval_batch_size=16": "per_device_eval_batch_size=1",
            "gradient_checkpointing=False": "gradient_checkpointing=True",
            "report_to=None": 'report_to="none"',
        }

        updated_line = line
        for old, new in replacements.items():
            updated_line = updated_line.replace(old, new)

        if "gradient_checkpointing=True" in updated_line:
            updated_source.append(updated_line)
            accumulation_line = "    gradient_accumulation_steps=16,\n"
            if accumulation_line not in source and accumulation_line not in updated_source:
                updated_source.append(accumulation_line)
            if updated_line != line:
                changed = True
            continue

        if updated_line != line:
            changed = True
        updated_source.append(updated_line)
    cell["source"] = updated_source

if changed:
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print("Updated SFTConfig for 12 GB VRAM and report_to=\"none\".")
else:
    print("No SFTConfig changes needed.")
PY

log "Creating Python 3.12 virtual environment"
if [[ "${RESET_ENV:-0}" == "1" ]]; then
  uv venv --python 3.12 --clear .venv
elif [[ -d .venv ]]; then
  log "Reusing existing .venv. Set RESET_ENV=1 to recreate it."
else
  uv venv --python 3.12 .venv
fi
source .venv/bin/activate

log "Installing CUDA PyTorch wheels"
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

log "Installing notebook dependencies"
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

log "Writing helper scripts"
cat > scripts/check_gpu.py <<'PY'
import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    x = torch.randn((2048, 2048), device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    used_mib = torch.cuda.max_memory_allocated() / 1024**2
    print(f"CUDA matmul OK: shape={tuple(y.shape)} dtype={y.dtype}")
    print(f"Peak allocated: {used_mib:.2f} MiB")


if __name__ == "__main__":
    main()
PY

cat > scripts/start_jupyter.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec jupyter lab \
  --ip="${JUPYTER_IP:-0.0.0.0}" \
  --port=8888 \
  --no-browser
SH
chmod +x scripts/start_jupyter.sh

log "Verifying PyTorch CUDA"
python scripts/check_gpu.py

log "Checking public dataset download"
python - <<'PY'
from datasets import load_dataset

dataset = load_dataset("mrdbourke/FoodExtract-1k", split="train[:3]")
print(dataset)
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  log "Logging into Hugging Face from HF_TOKEN"
  hf auth login --token "$HF_TOKEN"
fi

log "Checking gated Gemma model access"
python - <<'PY'
from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
except Exception as exc:
    print("Gemma access check failed.")
    print("Accept access at https://huggingface.co/google/gemma-3-270m-it and run:")
    print("  source .venv/bin/activate && hf auth login")
    print(f"{type(exc).__name__}: {str(exc)[:500]}")
else:
    print(f"Gemma tokenizer OK: {tokenizer.__class__.__name__}")
PY

if [[ "${START_JUPYTER:-0}" == "1" ]]; then
  log "Starting JupyterLab"
  exec ./scripts/start_jupyter.sh
fi

log "Setup complete"
cat <<EOF

Workspace:
  $WORKSPACE

Open the notebook:
  $WORKSPACE/$NOTEBOOK_FILE

Start Jupyter:
  cd "$WORKSPACE"
  ./scripts/start_jupyter.sh

Tailscale URL:
  http://100.92.158.40:8888

If Gemma access failed:
  1. Accept terms at https://huggingface.co/google/gemma-3-270m-it
  2. Run: source .venv/bin/activate && hf auth login

EOF
