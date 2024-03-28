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
(apollo) /home/pp23/share/hw3-3

![3-1 source](/assignments/hw3%20All-Pairs%20Shortest%20Path/images/3-1_source.png)