- seq.cc
    > Blocked Floyd-Warshall Algorithm 的循序版本
- hw3-2_gm.cu (global memory version)
    > phase 1,2,3 皆不使用 shared memory，參考與更改 entries 的值都是直接對 global memory 的 d_dist_matrix 進行操作
- hw3-2_sm.cu (shared memory version)
    > 在 phase 1,2,3 中，會在運算前先將稍後運算會參考到的 entris 值存至 shared memory 中，而運算時參考與更改 entries 值都是對於 shared memory 的操作，運算結束後才將最終 entry 值從 shared memory 存回 global memory 的 d_dist_matrix 對應位置。
- hw3-2_bb.cu (big block version)
    > 概念上和 shared memory version 相同，差別在於為了充分利用一個 device block 的 shared memory 最大容量 (49152 = 3＊64＊64 bytes)，blocking factor 會設定為 64，因此在 phase 1,2,3 中每條 thread 一次可以處理 4 entries
- hw3-2_bb_time.cu
    > 基於 hw3-2_bb.cu，在 input, output 部分利用 omp_get_wtime() 來計時，和 device 端相關的操作則利用 cudaEvent 來計時。
- profile.sh
    > 裡面的這些指令可以根據你選擇的 metrics 去呈現 device 端在運行時的各項資訊
    > 
    > ex. 欲呈現測試 c15.1 時 ipc, achieved_occupancy, sm_efficiency 這三項 metrics 的結果
    > 
    > $ nvprof --metrics ipc,achieved_occupancy,sm_efficiency ./hw3-2 /home/pp23/share/hw3-2/cases/c15.1 ./c15.1.out ./profc15.log