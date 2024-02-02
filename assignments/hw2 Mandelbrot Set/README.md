## Description
本次作業為利用 Pthread、OpenMP、MPI 等方式來平行化 Mandelbrot Set 的計算，以減少程式執行時間，並透過 Vectorization 來作更進一步的優化。下圖說明為 Mandelbrot Set 的介紹與計算方式，測資會給定複座標的橫軸與縱軸範圍 ($x0,$x1,$y0,$y1)，以及輸出圖片的大小 ($w,$h)，work() 會先計算圖片中每個 pixel 代表的迭代計算結果 pixel_value，write_png() 再將所有 pixel_value 對應成顏色來輸出成圖片，即可得到 Mandelbrot Set 的圖形。

![problem](/assignments/hw2%20Mandelbrot%20Set/images/problem.png)
## Result
## 檔案說明
- hw2
    > 此資料夾即為我在工作目錄下創建的 hw2 資料夾，裡面包含了 hw2a (Pthread version)、hw2b (Hybrid version)、hw2seq (Sequential version) 這幾個不同的實作方式。
- images
    > 存放此 repo 中所有 README.md 用到的圖片。
- slides & announcement
    > 此資料夾包含本次作業的說明文件、補充簡報，以及相關章節的上課講義。
## URL
- [hackmd: report](https://hackmd.io/@u_46AznXS7-aLzZ7_uD4WQ/BJHnOwv96)
- [hw2a scoreboard](https://apollo.cs.nthu.edu.tw/pp23/scoreboard/hw2a/)、[hw2b scoreboard](https://apollo.cs.nthu.edu.tw/pp23/scoreboard/hw2b/)
- [hw2 score - pp23s90](https://docs.google.com/spreadsheets/d/1JnFx8Byu1UGUygVXx1_bmjnZ2_kysicBdxEbUeFIY8E/edit?usp=sharing)、[hw2 score - public](https://docs.google.com/spreadsheets/d/1eXI1YN410rll8yjRZyN8wMEU4HGsUhjRR-PRSj-_m8k/edit?usp=sharing)