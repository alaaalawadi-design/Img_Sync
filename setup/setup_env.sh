#!/bin/bash
set -x
source /home/administrator/miniconda3/etc/profile.d/conda.sh
ENV=/home/administrator/miniconda3/envs/longcat-avatar15
REPO=/home/administrator/imgsync/LongCat-Video
SRCW=/home/administrator/alaa-workspace/LongCat-Video/weights
# --- weights layout: verified files only (base text-to-video dit/ excluded) ---
mkdir -p $REPO/weights/LongCat-Video
for f in config.json model_index.json scheduler text_encoder tokenizer vae; do ln -sfn $SRCW/LongCat-Video/$f $REPO/weights/LongCat-Video/$f; done
ln -sfn $SRCW/LongCat-Video-Avatar-1.5 $REPO/weights/LongCat-Video-Avatar-1.5
ls -la $REPO/weights $REPO/weights/LongCat-Video
# --- env ---
[ -d $ENV ] || conda create -y -p $ENV python=3.10
conda activate $ENV
python -V
# RTX 5090 (sm_120) requires CUDA>=12.8 builds; official torch 2.6/cu124 has no sm_120 kernels
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install ninja psutil packaging
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
grep -vE '^(torch|flash-attn)==' $REPO/requirements.txt > /tmp/req_lc.txt
pip install -r /tmp/req_lc.txt
grep -vE '^(libsndfile1|tritonserverclient|openai)==' $REPO/requirements_avatar.txt > /tmp/req_av.txt
pip install -r /tmp/req_av.txt
pip install "huggingface_hub[cli]"
python - <<'PY'
import torch, flash_attn, transformers, diffusers, librosa, audio_separator
print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("flash_attn", flash_attn.__version__, "transformers", transformers.__version__, "diffusers", diffusers.__version__)
from flash_attn import flash_attn_func
q = torch.randn(1, 4096, 32, 128, device="cuda", dtype=torch.bfloat16)
o = flash_attn_func(q, q, q)
ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2), q.transpose(1,2), q.transpose(1,2)).transpose(1,2)
print("flash_attn sm_120 OK, max diff vs SDPA:", (o-ref).abs().max().item())
PY
echo SETUP_DONE
