#! /bin/bash


python gmcnn/training/gmcnn/train.py --mask_type "custom" --load_model_dir "checkpoints/GMCNN/20220126-125206_GMCNN_kvasir_1000_b6_s256x256_gc32_dc64_randmask-stroke" --root_path "data/kvasir-seq-1000/Kvasir-SEG" --random_crop 0 --phase "train" --batch_size 6