# 安裝環境教學
https://medium.com/@EricChou711/nvidia-jetson-agx-orin-%E5%AE%8C%E6%95%B4%E5%88%B7%E6%A9%9F-%E5%AE%89%E8%A3%9D-tensorflow-pytorch-opencv-%E6%95%99%E5%AD%B8-ubuntu%E7%AF%87-sdk-manager-b3395f654f75

# Jetson-AGX-Orin 安裝環境
刷機完成，載入GPU驅動+JetPack 6.2.1 + CUDA + cuDNN + TensorRT
<img width="1748" height="860" alt="image" src="https://github.com/user-attachments/assets/240d5fe0-408b-44fe-b2dc-7589bd0e9736" />
# 遇到問題
(1) 由於jetPack 6.2.1沒有提供PyTorch GPU版，所以要啟用NVIDIA官方Docker(含PyTorch+CUDA)，
由於官方沒有yolo11 pose容器可以直接跑在jetson，所以自己建立image(yolo11-jetson)
之後再將專案掛進Docker裡面執行

(2)安裝pytroch到jetson主機要跟dockerfile放一起

下載torch-2.1.0-cp310-cp310-linux_aarch64.whl

(https://forums.developer.nvidia.com/t/pytorch-for-jetpack-6-0/275200)
```
pip3 install torch-2.1.0-cp310-cp310-linux_aarch64.whl
```
測試有無torch跟cuda
```
python3 - << EOF
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
EOF
```
(3)torchvision 安裝(因為 Jetson 沒有提供 wheel，所以 必須從 source 編譯)

安裝依賴
```
sudo apt install libjpeg-dev zlib1g-dev
```
下載 torchvision source
```
cd ~
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.16.0
```
編譯與安裝
```
cd ~/vision
sudo -E python3 setup.py install
```
測試
```
python3 - << EOF
import torchvision
print("torchvision:", torchvision.__version__)
EOF
```

拉取 NVIDIA 官方 PyTorch GPU 映像
```
sudo docker pull nvcr.io/nvidia/l4t-base:r36.2.0
```

## 建立image(yolo11-jetson)
建立dockerfile放在(cd ~/docker/yolo11)裡面
```
# ============================================================
# Jetson Orin YOLO11 + PyTorch 2.1.0 + Flask + WebSocket + RTSP
# Base Image: NVIDIA Jetson L4T Base (CUDA runtime auto-mount)
# ============================================================

FROM nvcr.io/nvidia/l4t-base:r36.2.0

ARG DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# 1. 安裝系統依賴 + GStreamer + Jetson OpenCV
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    build-essential \
    libopenblas-base \
    python3-opencv libopencv-dev \
    libglib2.0-0 libgl1-mesa-glx libgtk-3-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    ffmpeg \
    nano vim git curl wget \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 2. 固定 numpy 版本（避免破壞 OpenCV）
# ------------------------------------------------------------
RUN pip install --upgrade pip wheel setuptools
RUN pip install numpy==1.26.4

# ------------------------------------------------------------
# 3. 安裝 PyTorch 2.1.0 (Jetson 專用 wheel)
# ------------------------------------------------------------
COPY torch-2.1.0-cp310-cp310-linux_aarch64.whl /tmp/
RUN pip install /tmp/torch-2.1.0-cp310-cp310-linux_aarch64.whl

# 安裝 GPU 版 TorchVision (Jetson 必須從 source 編譯)
RUN git clone --branch v0.16.0 https://github.com/pytorch/vision.git /opt/vision
WORKDIR /opt/vision
RUN python3 setup.py install
# ------------------------------------------------------------
# 4. 安裝 YOLO11 依賴（不碰 OpenCV）
# ------------------------------------------------------------
ENV UV_DISABLE_OPENCV_IMPORT=1
RUN pip install ultralytics supervision --no-deps
RUN pip install onnx==1.14.1
RUN pip install onnxruntime==1.17.3 --no-deps

# 確保沒有 pip opencv 汙染 Jetson 內建 cv2
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true

# ------------------------------------------------------------
# 5. 你的專案依賴
# ------------------------------------------------------------
RUN pip install \
    Flask==3.1.2 \
    Werkzeug==3.1.3 \
    websockets==15.0.1 \
    pillow==10.2.0 \
    PyYAML==6.0.1 \
    requests==2.32.3 \
    psutil==5.9.8 \
    typing_extensions==4.7.1 \
    ffmpeg-python==0.2.0

# ------------------------------------------------------------
# 6. 容器工作路徑
# ------------------------------------------------------------
WORKDIR /workspace

# ------------------------------------------------------------
# 7. 預設進入 bash
# ------------------------------------------------------------
CMD ["/bin/bash"]
```
## 開始建置 Image
```
cd ~/docker/yolo11
sudo docker build --network host -t yolo11-jetson .
```
# 執行程式
jetson主機執行
```
xhost +
```
啟動yolo11容器
```
sudo docker run -it --rm \
  --runtime=nvidia \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/Desktop/Detect_MoveTrack:/workspace \
  --ipc=host \
  yolo11-jetson

```
進入容器後 
```
cd /workspace
```
容器內測試
```
python3 - << EOF
import torch, torchvision, cv2
print("Torch:", torch.__version__)
print("Vision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("OpenCV:", cv2.__version__)
EOF
```
執行
```
python3 main.py
```
