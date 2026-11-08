#!/bin/bash
set -x
source /home/administrator/miniconda3/etc/profile.d/conda.sh
conda activate /home/administrator/miniconda3/envs/longcat-avatar15
pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 "sympy==1.13.3" "numpy==1.26.4" --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
pip install --force-reinstall --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
pip check
pip list 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|sympy|numpy|flash|transformers|diffusers) '
python - <<'PY'
import torch, flash_attn, transformers, diffusers, librosa, audio_separator
print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())
print("flash_attn", flash_attn.__version__, "transformers", transformers.__version__, "diffusers", diffusers.__version__)
from flash_attn import flash_attn_func
q = torch.randn(1, 4096, 32, 128, device="cuda", dtype=torch.bfloat16)
o = flash_attn_func(q, q, q)
ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2), q.transpose(1,2), q.transpose(1,2)).transpose(1,2)
print("flash_attn sm_120 OK, max diff vs SDPA:", (o-ref).abs().max().item())
PY
echo FIX_DONE
