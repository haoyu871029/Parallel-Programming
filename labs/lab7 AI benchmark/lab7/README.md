## 檔案說明
- hades_2
    > 此資料夾內含一個用來在 hades02 上 fine-tune model 的腳本檔案 run_DDP.sh
- slurm_hades
    > 此資料夾內含兩個用來在 hades01 (會連到 hades03~07) 上 fine-tune model 的腳本檔案 run_DDP_1GPU.sh 和 run_DDP_2GPU.sh
- run_clm.py
    > 此 python 檔為模型訓練腳本，會在 run_DDP.sh, run_DDP_1GPU.sh, run_DDP_2GPU.sh 中被調用
- try_out.py
    > 此 python 檔用來和 fine-tuned model 互動