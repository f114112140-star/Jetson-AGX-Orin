# 安裝環境教學
https://medium.com/@EricChou711/nvidia-jetson-agx-orin-%E5%AE%8C%E6%95%B4%E5%88%B7%E6%A9%9F-%E5%AE%89%E8%A3%9D-tensorflow-pytorch-opencv-%E6%95%99%E5%AD%B8-ubuntu%E7%AF%87-sdk-manager-b3395f654f75

# Jetson-AGX-Orin 安裝環境
刷機完成，載入GPU驅動+JetPack 6.2.1 + CUDA + cuDNN + TensorRT
<img width="1748" height="860" alt="image" src="https://github.com/user-attachments/assets/240d5fe0-408b-44fe-b2dc-7589bd0e9736" />
# 遇到問題
由於jetPack 6.2.1沒有提供PyTorch GPU版，所以要啟用NVIDIA官方Docker(含PyTorch+CUDA)，
由於官方沒有yolo11 pose容器可以直接跑在jetson，所以自己建立image(yolo11-jetson)
之後再將專案掛進Docker裡面執行
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
