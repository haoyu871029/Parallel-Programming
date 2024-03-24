## Description
這次作業的目標是實作演算法來解決 All-Pairs Shortest Path Problem，再將程式做平行化，需分成三部分來實作：
- hw3-1: CPU Floyd-Warshall Algorithm
    > 我選擇實作的演算法是 Floyd-Warshall Algorithm，由於作業要求以 threading 的方式來作平行化，我選擇利用 OpenMP API 來平行化 sequential 版本的 Floyd-Warshall Algorithm
- hw3-2: Single-GPU Blocked Floyd-Warshall Algorithm
    > 透過 CUDA API，利用單 GPU 平行化 Blocked Floyd-Warshall Algorithm 的運算，我將 phase 1, phase 2, phase 3 分別寫成一個 kernel function 給 device 端處理，共實作了 global memory, shared memory, big block 三個版本。
- hw3-3: Multi-GPU Blocked Floyd-Warshall Algorithm
    > 透過 CUDA API，利用多 GPU 平行化 Blocked Floyd-Warshall Algorithm 的運算，我指定 device 數量為 2，並基於 hw3-2 的 big block 版本改寫 phase 2, phase 3，讓 devices 可以作分配。
## Result
## 檔案說明
- hw3
    > 此資料夾即為我在工作目錄下創建的 hw3 資料夾，裡面包含了各版本的實作、Makefile 等等。
- images
    > 存放此 repo 中所有 README.md 用到的圖片。
- slides & announcement
    > 此資料夾有作業公告以及作業原檔說明，並包含本次作業的說明文件(spec)、相關章節的上課講義與簡報。
## URL
- [hackmd: report](https://hackmd.io/@u_46AznXS7-aLzZ7_uD4WQ/rkguXJ3Fp)
- [3-1 scoreboard](https://apollo.cs.nthu.edu.tw/pp23/scoreboard/hw3-1/)、[3-2 scoreboard](https://apollo.cs.nthu.edu.tw/pp23/scoreboard/hw3-2/)、[3-3 scoreboard](https://apollo.cs.nthu.edu.tw/pp23/scoreboard/hw3-3/g)
- [hw3 score - pp23s90](https://docs.google.com/spreadsheets/d/1JnFx8Byu1UGUygVXx1_bmjnZ2_kysicBdxEbUeFIY8E/edit?usp=sharing)、[hw3 score - public](https://docs.google.com/spreadsheets/d/1_j22lcEnxnMS3oGOq0fRU_FMs7Pzzorkt_Aryic65yQ/edit?usp=sharing)