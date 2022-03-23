#! bin/bash


python aotgan/src/train.py --dir_image "data/kvasir-seq-1000/Kvasir-SEG" --mask_type "masks" --iterations 10000 --batch_size 2 --save_dir "finetuned_aotgan_checkpoints" --tensorboard