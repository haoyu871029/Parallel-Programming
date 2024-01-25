## Version
- hw3-3.cu
    > 指定 device 數為 2，並基於 hw3-2_bb.cu 改寫 phase 2, phase 3，讓 devices 可以作分配。(report 中描述的是只改寫 phase 3 的情況)
- hw3-3_time.cu
    > 基於 hw3-3.cu，利用 cudaEvent 紀錄各個 device 作 Blocked Floyd-Warshall Algorithm 所需的時間。

欲執行特定版本，先將該版本的程式碼複製到 hw3-3.cc，再編譯執行。
## Other files
- Makefile
    > (編譯 hw3-3.cc) $ make hw3-3
## 作業原檔
![截圖 2024-01-26 上午12.51.52](https://hackmd.io/_uploads/ryE2-Gl9a.png)