CUDA_VISIBLE_DEVICES=0,1 python train.py \
      --dataset 'miqa_cls' \
      --path_miqa_cls '/public/datasets/miqa_cls' \
      --train_split_file '../data/dataset_splitting/miqa_cls_train.csv' \
      --val_split_file '../data/dataset_splitting//miqa_cls_val.csv' \
      --metric_type 'composite' --loss_name 'mse' \
      -a 'vit_small_patch16_224' --pretrained --transform_type 'simple_transform' \
      -b 256 --epochs 5 --warmup_epochs 1 --validate_num 2 --lr 1e-4 \
      --image_size 288 --crop_size 224 --workers 8 -p 100 \
      --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0

