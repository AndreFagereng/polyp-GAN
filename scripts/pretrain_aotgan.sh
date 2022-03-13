#! bin/bash


python aotgan/src/train.py --dir_image "data/kvasir-seq-pretrain" --dir_mask "data/kvasir-seq-pretrain" --data_train "images" --data_test "" --image_size 512 --mask_type "irregular_mask" --iterations 500000 --batch_size 8 --save_dir "pretrained_aotgan_checkpoints_dgx" --tensorboard --outputs "outputs"

