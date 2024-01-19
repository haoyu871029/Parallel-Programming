## description
實作 Fiduccia-Mattheyses algorithm，並使用 openmp、pthread、mpi 進行平行化。

實作流程與平行化部分如下圖所示：

![figure](https://hackmd.io/_uploads/H1Z1sqvFp.png)
## version
- openmp.cpp (openmp)
    > 在 Fn(), Tn(), G(), update_gain() 等函式中加入 openmp API 作平行化。
- pthread.cpp (openmp + pthread)
    > 在 openmp.cpp 的基礎上，利用 pthread 在前期篩選出最好的 initoal cut size 來進行 Fiduccia-Mattheyses algorithm
- mpi.cpp (openmp + mpi)
    > 在 openmp.cpp 的基礎上，利用 mpi 篩選出最好的 final cut size 寫入 output file
## url
- [google slides: presentation](https://docs.google.com/presentation/d/149GhUorqxLvylHjvkFctsATc2HO3H4iDoY7Bdx6wEic/edit?usp=sharing)