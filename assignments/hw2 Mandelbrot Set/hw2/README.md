## 檔案說明
- hw2a
    > 此資料夾內包含了同樣使用 Pthread，但用不同方式來進行平行化的各版本程式
- hw2b
    > 此資料夾內包含了同樣是 Hybrid (MPI+OpenMP)，但用不同方式來進行平行化的各版本程式
- hw2a.cc
    > hw2a_vec_v2.cc in hw2a
- hw2b.cc
    > hw2b_vec_v2.cc in hw2b
- hw2seq.cc
    > sequential version
- Makefile
    > 可編譯 hw2a、hw2b、hw2seq 這三個程式的 Makefile
    > 
    > (compile hw2a.cc) $ make hw2a
    > 
    > (compile hw2b.cc) $ make hw2b
    > 
    > (compile hw2seq.cc) $ make hw2seq