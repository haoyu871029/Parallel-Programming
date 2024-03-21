## Version
- hw1_v1.cc
    > Odd-even sort 階段修正前的寫法
- hw1_v2.cc
    > 修正方法一，對應到實驗階段的 Ver_A (最終繳交版本)
- hw1_v3.cc
    > 修正方法二，對應到實驗階段的 Ver_B
- hw1.cc
    > 欲執行特定版本程式，先將程式碼複製到此檔案再編譯執行。
## Other files
- hw1_v2_time.cc
    > 在 hw1_v2.cc 中加入 MPI_Wtime() 來計算程式的各細項執行時間
- mpiio.cc
    > MPI-IO 範例程式碼，此程式可以用來測試：
    > - 從測資 .in 檔中對應位置讀取浮點數資料
    > - 將數據寫入 .out 檔中的對應位置
- Makefile
    > (編譯 hw1.cc) $ make