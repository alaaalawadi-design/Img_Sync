#!/bin/bash
set -x
source /home/administrator/miniconda3/etc/profile.d/conda.sh
E=/home/administrator/miniconda3/envs/avatar-eval
[ -d $E ] || conda create -y -p $E python=3.10
conda activate $E
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install "numpy<2" opencv-python-headless==4.10.0.84 scikit-image scipy python_speech_features scenedetect==0.5.6.1 lpips pyiqa psutil pillow
cd /home/administrator/imgsync/eval
M=models
wget -q -nc -O $M/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
wget -q -nc -O $M/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
[ -d vendor/syncnet_python ] || git clone --depth 1 https://github.com/joonson/syncnet_python vendor/syncnet_python
wget -q -nc -O $M/syncnet_v2.model http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model
ls -la $M
python - <<'PY'
import torch, torchvision, cv2, lpips, pyiqa
print("torch", torch.__version__, "cv2", cv2.__version__)
m = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT"); print("deeplab ok")
q = pyiqa.create_metric("musiq", device="cpu"); print("musiq ok")
l = lpips.LPIPS(net="alex"); print("lpips ok")
PY
echo EVAL_SETUP_DONE
