## Description
本次 lab 是 GPT-2 model 的 fine-tune 與使用。
- 藉由更改 .sh 腳本檔案中的 "--nproc_per_node" 參數值，來指定欲使用的 GPU 數量，以透過訓練資料的平行分配來加速訓練過程，最後觀察訓練結果 (train metrics)，可透過 try_out.py 來與模型互動。
- 若要在 hades01 進行測試，run_DDP_1GPU.sh 與 run_DDP_2GPU.sh 已分別指定好"--nproc_per_node" 參數值為 1 和 2，可直接分別執行以觀察不同 GPU 數量下訓練結果。若要在 hades02 進行測試，則需自行指定 run_DDP.sh 中的 "--nproc_per_node" 參數值來進行測試
## URL
- [hackmd: report](https://hackmd.io/@u_46AznXS7-aLzZ7_uD4WQ/HkwjzDRt6)
## 待辦
- announcement 的截圖