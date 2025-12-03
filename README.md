# 安裝環境教學
https://medium.com/@EricChou711/nvidia-jetson-agx-orin-%E5%AE%8C%E6%95%B4%E5%88%B7%E6%A9%9F-%E5%AE%89%E8%A3%9D-tensorflow-pytorch-opencv-%E6%95%99%E5%AD%B8-ubuntu%E7%AF%87-sdk-manager-b3395f654f75

# Jetson-AGX-Orin 安裝環境
刷機完成，載入GPU驅動+JetPack 6.2.1 + CUDA + cuDNN + TensorRT
<img width="1748" height="860" alt="image" src="https://github.com/user-attachments/assets/240d5fe0-408b-44fe-b2dc-7589bd0e9736" />
# 遇到問題
由於jetPack 6.2.1沒有提供PyTorch GPU版，所以要啟用NVIDIA官方Docker(含PyTorch+CUDA)，
由於官方沒有yolo11 pose容器可以直接跑在jetson，所以自己建立image(yolo11-jetson)
之後再將專案掛進Docker裡面執行

拉取 NVIDIA 官方 PyTorch GPU 映像
```
nvcr.io/nvidia/pytorch:24.06-py3
```
這是 JetPack 6.2.1（CUDA 12.2）唯一支援 GPU 的 PyTorch。

成功啟動 PyTorch GPU 容器
```
sudo docker run -it --rm --runtime=nvidia nvcr.io/nvidia/pytorch:24.06-py3
```
## 建立image(yolo11-jetson)
建立dockerfile放在(cd ~/docker/yolo11)裡面
```
# ============================================================
# Jetson Orin YOLO + PyTorch + Flask + WebSocket + RTSP 開發環境
# Base Image: NVIDIA PyTorch 24.06 (CUDA 12.x, JetPack 6.x)
# ============================================================

FROM nvcr.io/nvidia/pytorch:24.06-py3

ARG DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# 1. 安裝系統工具 & GStreamer（RTSP 需要）
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
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

RUN pip install --upgrade pip wheel setuptools

# ------------------------------------------------------------
# 2. 安裝 YOLO11 + 推論相關工具
# ------------------------------------------------------------
RUN pip install ultralytics supervision onnx onnxruntime

# ------------------------------------------------------------
# 3. 安裝你的 requirements.txt 所需套件
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
# 4. 複製你的 Detect_MoveTrack 專案
# ------------------------------------------------------------
WORKDIR /workspace
COPY . /workspace

ENV PYTHONPATH=/workspace:$PYTHONPATH

# ------------------------------------------------------------
# 5. 預設執行 main.py
# ------------------------------------------------------------
CMD ["python3", "main.py"]
```
## 開始建置 Image
```
cd ~/docker/yolo11
sudo docker build -t yolo11-jetson .
```
# 執行程式
進入資料夾，並啟動yolo11容器

cd ~/Desktop/Detect_MoveTrack

sudo docker run --runtime=nvidia -it --rm \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/Desktop/Detect_MoveTrack:/workspace \
  --network host \
  yolo11-jetson

進入容器後 cd /workspace

執行 python3 main.py
