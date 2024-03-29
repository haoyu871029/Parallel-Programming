## Description
主題與 lab3 相同，透過 sobel operator 來做 image edge detection。而本次 lab 的目標，是透過實作 coalesced memory, lower precision, shared memory 等方式，去修改 lab3 實作的 CUDA program 或 TA 提供的 lab4 source code，以達到進一步的優化。

更詳細的作業說明請參考 lab4_spec.pdf
## Result
![result](/labs/lab4%20CUDA%20Advance/images/result.png)
- Coalesced Memory
    > 將 block dimension 從原本的 16x16 改為 32x32，讓同一個 warp 中的 threads 存取連續的記憶體位置，以合併記憶體存取。
- Lower Precision
    > 將 RGB 計算值的資料型態從原本的 double 改為 float，雖然降低了精度，但能在不影響正確性的情況下減少計算時間。

雖然還沒有實作 shared memory，但透過上述兩種方式做修正，已經能將程式的執行時間降低。
## 檔案說明
- lab4
    > 此資料夾即為我在工作目錄下創建的 lab4 資料夾，包含 lab3_bb.cu, lab4_cm.cu, lab4_source.cu, Makefile 等檔案。
- slides & announcement
    > 此資料夾包含本次 lab 的說明簡報以及相關的上課講義。
- images
    > 存放此 repo 中所有 README.md 用到的圖片。
## URL
- [note (hackmd)](https://hackmd.io/@u_46AznXS7-aLzZ7_uD4WQ/S1Sts9X26)
- [lab4 scoreboard](https://apollo.cs.nthu.edu.tw/pp23/scoreboard/lab4/)
- [lab4 slides](https://docs.google.com/presentation/d/1AH-5ZQ32tTVJh7zge4Sqej6O6GhY8Gz-0iqZQ8exyNM/edit?usp=sharing)
- [NCHC Webpage](https://portal.apps.edu-cloud.nchc.org.tw), [NCHC Tutorial](https://hackmd.io/@enmingw32/pp-nchc)