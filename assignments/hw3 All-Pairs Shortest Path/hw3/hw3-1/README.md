## Version
- seq.cc
    > Floyd-Warshall Algorithm 的循序版本
- openmp.cc
    > 基於 seq.cc，利用 OpemMP API 在 "build default matrix" 以及 "FW()" 兩處作平行化
- openmp_time.cc
    > 在 openmp.cc 中利用 omp_get_wtime() 計算各步驟的執行時間

欲執行特定版本，先將該版本的程式碼複製到 hw3-1.cc，再編譯執行。
## Other files
- Makefile
    > (編譯 hw3-1.cc) $ make hw3-1
## 作業原檔
![截圖 2024-01-26 上午12.53.26](https://hackmd.io/_uploads/B14fGGxqp.png)