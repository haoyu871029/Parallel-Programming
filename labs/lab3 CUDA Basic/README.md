## Description
本次 lab 的主題是透過 sobel operator 來做 image edge detection，其中 sobel operator 的處理涉及 convolution calculation，image 中的 每個 pixel 會與兩個 filter matrixs 進行矩陣運算，如下圖所示。而實作的目標為，設計 CUDA programming 來平行化此部分的運算，以加速處理速度。

![desc](/labs/lab3%20CUDA%20Basic/images/desc.png)
- Edge Detection
    > Identifying points in a digital image at which the image brightness changes sharply
- Sobel Operator
    > Used in image processing and computer vision, particularly within edge detection algorithms. Uses two 3x3 filter matrix gx, gy (one for horizontal changes, and one for vertical) which are convolved with the original image to calculate approximations of the derivatives. In this lab, we use 5x5 kernels.
- Convolution Calculation
    > Iterate through the width and height of the image. For each pixel, multiply the filter matrix with original image element-wisely and sum them up.

更詳細的作業說明請參考 lab3_spec.pdf
## Result
![resu](/labs/lab3%20CUDA%20Basic/images/resu.png)

將 sobel operator 的計算交給 device 端分配給 threads 來執行後，各測資的程式執行時間相較於 cpu version 皆顯著下降，而 filter matrix 若置於 constant memory 來存取，結果又略好於將其置於 local memory
## 檔案說明
- lab3
    > 此資料夾即為我在工作目錄下創建的 lab3 資料夾，包含 cpu version code, gpu version code, Makefile 等檔案。
- slides & announcement
    > 此資料夾包含本次 lab 的說明簡報以及相關的上課講義。
- images
    > 存放此 repo 中所有 README.md 用到的圖片。
## URL
- [note (hackmd)](https://hackmd.io/@u_46AznXS7-aLzZ7_uD4WQ/ryIWGdqia)
- [lab3 scoreboard](https://apollo.cs.nthu.edu.tw/pp23/scoreboard/lab3/)
- [NCHC Webpage](https://portal.apps.edu-cloud.nchc.org.tw), [NCHC Tutorial](https://hackmd.io/@enmingw32/pp-nchc)