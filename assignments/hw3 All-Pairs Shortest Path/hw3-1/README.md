- seq.cc
    > Floyd-Warshall Algorithm 的循序版本
- openmp.cc
    > 基於 seq.cc，利用 opemMP API 在 "build default matrix" 以及 "FW()" 兩處作平行化
- openmp_time.cc
    > 在 openmp.cc 中利用 omp_get_wtime() 計算各步驟的執行時間