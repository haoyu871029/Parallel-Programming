## Description
UCX 實作/支援 MPI processes 之間的溝通行為，本次作業的重點是透過 trace UCX system code 來觀察 UCP Objects（Context, Worker, End point）之間的函式呼叫關係，以認識 UCP Objects 彼此之間是如何互動來完成程式中 communication 的部分（資料傳遞與接收），實作部分則是關於 UCX_TLS（通訊方式）的選擇。

UCP Objects 之間如何透過函式呼叫來互動，整理如下圖所示：(主程式為 ucp_hello_world.c)

![architecture](/assignments/hw4%20Observe%20the%20behavior%20of%20UCX%20yourself/images/des.png)
## 檔案說明
- images
    > 存放此 repo 中所有 README.md 用到的圖片。
- slides & announcement
    > 此資料夾有作業公告以及作業原檔說明，並包含本次作業的說明文件(spec)、相關章節的上課講義與簡報。
## URL
- [hackmd: report](https://hackmd.io/@u_46AznXS7-aLzZ7_uD4WQ/SkclAKwK6)
- [online tool: markdown to pdf](https://md2pdf.netlify.app/)
- [HW4 scoreboard](https://apollo.cs.nthu.edu.tw/pp23/scoreboard/hw4/)
- [hw4 score - pp23s90](https://docs.google.com/spreadsheets/d/1JnFx8Byu1UGUygVXx1_bmjnZ2_kysicBdxEbUeFIY8E/edit?usp=sharing)、[hw4 score - public](https://docs.google.com/spreadsheets/d/1_tlAxMmPNZtAyxAvnj5Jn81Ez1vIqAUP_tz5d_SSduA/edit?usp=sharing)