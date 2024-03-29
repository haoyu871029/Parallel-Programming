## Description
本次作業的應用是，若要填滿一個半徑為 r 的圓，需要多少個 pixels，詳細介紹及計算方式如下圖說明。而我們的實作目標就是利用 MPI APIs 去改寫原本的 sequential code，將計算部分分配給 processes 去處理，以減少程式的執行時間。

![description_1](/labs/lab1%20Platform%20Introduction%20&%20MPI/images/description_1.png)
![description_2](/labs/lab1%20Platform%20Introduction%20&%20MPI/images/description_2.png)

更詳細的作業說明請參考 lab1_spec.pdf
## Result
![result](/labs/lab1%20Platform%20Introduction%20&%20MPI/images/result.png)

由三筆測資的測試結果可以看出，透過 MPI APIs 的使用，我們將計算部分分配給 processes 去處理，結果也大幅降低了程式的執行時間。
## 檔案說明
- lab1
    > 此資料夾即為我在工作目錄下創建的 lab1 資料夾，包含 basic mpi code (hello.c), sequential code (lab1_source.cc), mpi version code (lab1_mpi.cc), Makefile 等檔案。
- slides & announcement
    > 此資料夾包含本次 lab 的說明簡報以及相關的上課講義。
- images
    > 存放此 repo 中所有 README.md 用到的圖片。
## URL
- [note (hackmd)](https://hackmd.io/@u_46AznXS7-aLzZ7_uD4WQ/rkoecl_aT)
- [lab1 scoreboard](https://apollo.cs.nthu.edu.tw/pp23/scoreboard/lab1/)
- [cluster monitor](http://apollo.cs.nthu.edu.tw/monitor)