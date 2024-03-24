## Version
- openmp.cpp (openmp)
    > 在 Fn(), Tn(), G(), update_gain() 等函式中加入 openmp API 作平行化。
- pthread.cpp (openmp + pthread)
    > 在 openmp.cpp 的基礎上，利用 pthread，在前期篩選出最小 initial cut size 的 thread 配置去作 Fiduccia-Mattheyses algorithm
- mpi.cpp (openmp + mpi)
    > 在 openmp.cpp 的基礎上，利用 mpi，在最後篩選出最小的 final cut size 寫入 output file

欲執行特定版本，先將該版本的程式碼複製到 main.cpp，再編譯執行。