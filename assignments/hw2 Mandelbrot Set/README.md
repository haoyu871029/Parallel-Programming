## Description
本次作業為利用 Pthread、OpenMP、MPI 等方式來平行化 Mandelbrot Set 的計算，以減少程式執行時間，並透過 Vectorization 來作更進一步的優化。下圖說明為 Mandelbrot Set 的介紹與計算方式，測資會給定複座標的橫軸與縱軸範圍 ($x0,$x1,$y0,$y1)，以及輸出圖片的大小 ($w,$h)，work() 會先計算圖片中每個 pixel 代表的迭代計算結果 pixel_value，write_png() 再將所有 pixel_value 對應成顏色來輸出成圖片，即可得到 Mandelbrot Set 的圖形。

![problem](/assignments/hw2%20Mandelbrot%20Set/images/des.png)
## Result
測資為 10000 -2 2 -2 2 800 800，以下呈現主要實驗結果，其他實驗結果在 report 內描述。

![ss](/assignments/hw2%20Mandelbrot%20Set/images/s_s.png)

Pthread version (左) 與 Hybrid version (右) 都有做出不錯的 Strong Scalibility

![rt](/assignments/hw2%20Mandelbrot%20Set/images/rt.png)

在測資不大的情況下，Hybrid version 的總執行時間較 Pthread version 長，而 Vectorization 確實可以降低總執行時間，但效果隨著 threads 數量增多而越不明顯。
## 檔案說明
- hw2
    > 此資料夾即為我在工作目錄下創建的 hw2 資料夾，裡面包含了 hw2a (Pthread version)、hw2b (Hybrid version)、hw2seq (Sequential version) 這幾個不同的實作方式。
- images
    > 存放此 repo 中所有 README.md 用到的圖片。
- slides & announcement
    > 此資料夾有作業公告以及作業原檔說明，並包含本次作業的說明文件(spec)、相關章節的上課講義與簡報。
## URL
- [hackmd: report](https://hackmd.io/@u_46AznXS7-aLzZ7_uD4WQ/BJHnOwv96)
- [hw2a scoreboard](https://apollo.cs.nthu.edu.tw/pp23/scoreboard/hw2a/)、[hw2b scoreboard](https://apollo.cs.nthu.edu.tw/pp23/scoreboard/hw2b/)
- [hw2 score - pp23s90](https://docs.google.com/spreadsheets/d/1JnFx8Byu1UGUygVXx1_bmjnZ2_kysicBdxEbUeFIY8E/edit?usp=sharing)、[hw2 score - public](https://docs.google.com/spreadsheets/d/1eXI1YN410rll8yjRZyN8wMEU4HGsUhjRR-PRSj-_m8k/edit?usp=sharing)
## 待辦
- 整理實驗結果 (多做 hw2b_vec 相關的實驗)