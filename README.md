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
# Base Image: NVIDIA Jetson PyTorch 2.1 (JetPack 6.x / CUDA 12.2)
# ============================================================

FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1

ARG DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# 1. 系統依賴 + GStreamer（RTSP 需要） + Jetson 原生 OpenCV
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
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
# 2. 升級 pip 並鎖定 numpy (避免 numpy2.x 導致 cv2 崩潰)
# ------------------------------------------------------------
RUN pip install --upgrade pip wheel setuptools
RUN pip uninstall -y numpy || true
RUN pip install numpy==1.26.4

# ------------------------------------------------------------
# 3. 安裝 YOLO11 + 推論工具
#    (禁止 pip OpenCV，避免覆蓋 Jetson 原生 OpenCV)
# ------------------------------------------------------------
ENV UV_DISABLE_OPENCV_IMPORT=1
RUN pip install ultralytics supervision onnx onnxruntime

# 強制移除 pip opencv（確保使用 Jetson 內建 cv2）
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true

# ------------------------------------------------------------
# 4. 安裝你的 requirements.txt 套件
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
# 5. 複製你的 Detect_MoveTrack 專案
# ------------------------------------------------------------
WORKDIR /workspace
COPY . /workspace
ENV PYTHONPATH=/workspace:$PYTHONPATH

# ------------------------------------------------------------
# 6. 預設進入 bash（不啟動 main.py）
# ------------------------------------------------------------
CMD ["/bin/bash"]
```
## 開始建置 Image
```
cd ~/docker/yolo11
sudo docker build -t yolo11-jetson .
```
# 執行程式
啟動yolo11容器
```
sudo docker run -it --rm \
  --gpus all \
  --runtime=nvidia \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/Desktop/Detect_MoveTrack:/workspace \
```
進入容器後 
```
cd /workspace
```
執行
```
python3 main.py
```
