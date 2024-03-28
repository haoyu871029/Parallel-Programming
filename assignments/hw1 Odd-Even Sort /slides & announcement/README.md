## Announcement
![announcement](/assignments/hw1%20Odd-Even%20Sort%20/images/announcement.png)
## 作業原檔
(apollo) /home/pp23/share/hw1

![source](/assignments/hw1%20Odd-Even%20Sort%20/images/source.png)
- testcases
    > 此為測資資料夾，其中的：
    > - .in 檔代表欲排序的浮點數資料，可以用 mpiio.cc 來讀取作測試。
    > - .out 檔代表預期的排序結果，可以將 hw1.cc 的輸出結果和此檔案比對確認正確性。
    > - .txt 檔代表該測資的說明，包含欲排序的浮點數總數(n)、評估用的資源量(nodes,procs)、規定的執行時間(time)
- Makefile
    > 用來編譯 hw1.cc，複製到工作目錄後沒變動過。
- mpiio.cc
    > MPI-IO 範例程式碼，可以用來讀測資，此檔案複製到工作目錄後僅多了測試用的 printf 以及編譯執行的指令提示。