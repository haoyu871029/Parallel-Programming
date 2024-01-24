## Description
實作 Fiduccia-Mattheyses algorithm，並使用 openmp、pthread、mpi 進行平行化。

實作流程與平行化部分如下圖所示：

![figure](https://hackmd.io/_uploads/H1Z1sqvFp.png)
## Version
- openmp.cpp (openmp)
    > 在 Fn(), Tn(), G(), update_gain() 等函式中加入 openmp API 作平行化。
- pthread.cpp (openmp + pthread)
    > 在 openmp.cpp 的基礎上，利用 pthread，在前期篩選出最小 initial cut size 的 thread 配置去作 Fiduccia-Mattheyses algorithm
- mpi.cpp (openmp + mpi)
    > 在 openmp.cpp 的基礎上，利用 mpi，在最後篩選出最小的 final cut size 寫入 output file
## URL
- [google slides: presentation](https://docs.google.com/presentation/d/149GhUorqxLvylHjvkFctsATc2HO3H4iDoY7Bdx6wEic/edit?usp=sharing)
- [hackmd: final project 想法與筆記](https://hackmd.io/@u_46AznXS7-aLzZ7_uD4WQ/SJZjPjwF6)