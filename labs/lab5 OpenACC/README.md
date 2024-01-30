## Description
DNN model 被廣泛應用在各個領域，例如影像辨識、語音辨識、自然語言處理等。在本次 lab 中，DNN model 被用來辨識手寫數字（handwritten digits），下圖為我們 DNN model 的訓練流程，模型架構為 2 fully-connected layer：

![DNN](/labs/lab5%20OpenACC/images/DNN.png)

相關資訊如下：
- Input dimension: 28x28
- Batch size: 60,000
- 3 types of calculation
    - Single Layer (y = aX + b)
    - Sigmoid
    - Argmax
- 2 fully-connected layers
    - Layer1: Linear layer + Sigmoid(activation function)
    - Layer2: Linear layer + Argmax

而我們的目標是使用 OpenACC Directive 來撰寫 GPU 程式碼，做到平行化來加速 DNN model 的訓練過程。
## 檔案說明
- lab5
    > 此資料夾即為我在工作目錄下創建的 lab5 資料夾，包含 seq.cpp、openacc.cpp、Makefile 等檔案。
- slides & announcement
    > 此資料夾包含本次 lab 的說明簡報以及 OpenACC 的教學投影片。
- images
    > 存放此 repo 中所有 README.md 用到的圖片。
## URL
- [OpenACC tutorial recording](https://drive.google.com/file/d/1yOeGrGYNzIiuozjlcMSrmV3DDbeJ0-nu/view)
- [hackmd: note](https://hackmd.io/@u_46AznXS7-aLzZ7_uD4WQ/BJ6hXwaI6)