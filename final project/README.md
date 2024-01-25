## Description
Fiduccia-Mattheyses Algorithm 用來解決 2-Way Partition Problem，目的是在滿足兩邊 subset 各自平衡條件的情況下，將 cells 分配到兩個 subsets 的同時最小化 subsets 之間的 nets 數 (minimize the cut size)

本專案先實作出 sequential 版本的 Fiduccia-Mattheyses Algorithm，再利用 OpenMP, Pthread, MPI 等方式將程式平行化，目標是降低程式執行時間與優化最終結果，實作流程與平行化部分如下圖所示：

![figure](https://hackmd.io/_uploads/H1Z1sqvFp.png)
## URL
- [google slides: presentation](https://docs.google.com/presentation/d/149GhUorqxLvylHjvkFctsATc2HO3H4iDoY7Bdx6wEic/edit?usp=sharing)
- [hackmd: final project 想法與筆記](https://hackmd.io/@u_46AznXS7-aLzZ7_uD4WQ/SJZjPjwF6)
## 待辦
- 補幾個測資