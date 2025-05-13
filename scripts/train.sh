#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate SAFE 

export CUDA_VISIBLE_DEVICES=1

GPU_NUM=1
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12588

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

train_datasets=(
    
    "/home/yiruolei/ALLDATASET/AIGCDetect/CNNSpot/train"\
)
eval_datasets=(
    "/home/yiruolei/ALLDATASET/AIGCDetect/CNNSpot/val" \
)
#"/home/yiruolei/ALLDATASET/GenImage" \
info="patch&highfreq三支路/backbone后cat/1-lowfreq+{mask一般掩码}"
#这里都是有logging信息的所以不需要自己nohup了

for train_dataset in "${train_datasets[@]}" 
do
    for eval_dataset in "${eval_datasets[@]}" 
    do

        current_time=$(date +"%Y%m%d_%H%M%S")
        OUTPUT_PATH="results/$info/$current_time"
        mkdir -p $OUTPUT_PATH

        python -m torch.distributed.run $DISTRIBUTED_ARGS main_finetune.py \
            --input_size 256 \
            --transform_mode 'ori' \
            --data_path "$train_dataset" \
            --eval_data_path "$eval_dataset" \
            --save_ckpt_freq 1 \
            --batch_size 64 \
            --blr 1e-2 \
            --weight_decay 0.01 \
            --warmup_epochs 1 \
            --epochs 15 \
            --num_workers 8 \
            --output_dir $OUTPUT_PATH \
        2>&1 | tee -a $OUTPUT_PATH/log_train.txt 

    done
done

#--transform_mode 'crop' \
#--batch_size 32 \
#num_workers 16 \