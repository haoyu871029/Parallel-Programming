## Description
本次作業的目標為實作出 odd-even sort algorithm，並利用 MPI APIs 將運算作平行處理，以降低程式的執行時間，並在分析階段使用 profile tool 來評估程式效能與平行化成果，關於 odd-even sort 的說明及例子如下所示。

Odd-even sort is a comparison sort that consists of two main phases: even-phase and odd-phase. 
In each phase, processes perform compare-and-swap operations repeatedly as follows until the input array is sorted.

Simple case：

![simple_case](/assignments/hw1%20Odd-Even%20Sort%20/images/simple_case.png)

更詳細的作業說明、實作要求與限制、報告規定等，請參考 hw1_spec.pdf
## Result
![res](/assignments/hw1%20Odd-Even%20Sort%20/images/res.png)

Ver_A: hw1_v2.cc 

Ver_B: hw1_v3.cc

如上圖所示，在固定 node 數的情況下，隨著運行的 process 數增加，兩個版本的程式執行時間皆有呈現減少的趨勢，具 strong scalability
## 檔案說明
- hw1
    > 此資料夾即為我在工作目錄下創建的 hw1 資料夾，裡面包含了各版本的實作檔、測資讀取範例程式、Makefile 等等。
- images
    > 存放此 repo 中所有 README.md 用到的圖片。
- slides & announcement
    > 此資料夾有作業公告以及作業原檔說明，並包含本次作業的說明文件(spec)、相關章節的上課講義與簡報。
## URL
- [report (hackmd)](https://hackmd.io/@u_46AznXS7-aLzZ7_uD4WQ/BJpWJ-g06)
- [hw1 scoreboard](https://apollo.cs.nthu.edu.tw/pp23/scoreboard/hw1/)
- [hw1 score - pp23s90](https://docs.google.com/spreadsheets/d/1JnFx8Byu1UGUygVXx1_bmjnZ2_kysicBdxEbUeFIY8E/edit?usp=sharing)、[hw1 score - public](https://docs.google.com/spreadsheets/d/1cltzY9Q27vwISqdnkgaXHOHMvuxJmBcrBPXhQuXs63U/edit#gid=0)