CUDA_VISIBLE_DEVICES=0,1 python train.py \
      --dataset 'miqa_ins' \
      --path_miqa_det '/public/datasets/miqa_det' \
      --path_miqa_ins '/public/datasets/miqa_ins/labels' \
      --train_split_file '../data/dataset_splitting/miqa_det_train.csv' \
      --val_split_file '../data/dataset_splitting//miqa_det_val.csv' \
      --metric_type 'composite' --loss_name 'mse' --is_two_transform \
      -a 'RA-MIQA' --pretrained --transform_type 'simple_transform' \
      -b 256 --epochs 5 --warmup_epochs 1 --validate_num 2 --lr 1e-4 \
      --image_size 288 --crop_size 224 --workers 8 -p 100 \
      --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0

