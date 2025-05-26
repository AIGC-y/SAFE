#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate SAFE 

export CUDA_VISIBLE_DEVICES=0

GPU_NUM=1
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12588  #原始是12588，修改以下

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
## 训练的时候是取出子目录所有数据,而不分每个小的来算
train_datasets=(
    
    # "/home/yiruolei/ALLDATASET/AIGCDetect/CNNSpot/train"\
    # "/home/yiruolei/ALLDATASET/AIGCDetect/Chameleon/test" \
    "/home/yiruolei/ALLDATASET/AIGCDetect/imagenet_ai_0419_sdv4/train"\
)
eval_datasets=(
    # "/home/yiruolei/ALLDATASET/AIGCDetect/CNNSpot/val" \
    # "/home/yiruolei/ALLDATASET/AIGCDetect/Chameleon/test" \
    "/home/yiruolei/ALLDATASET/AIGCDetect/imagenet_ai_0419_sdv4/val"\
)
ratio_data=0.8
#"/home/yiruolei/ALLDATASET/GenImage" \
info="sdv4test/safe结构测试/vit结构原始测试"

#****先把low的也尝试了，然后换resnet
#这里都是有logging信息的所以不需要自己nohup了

for train_dataset in "${train_datasets[@]}" 
do
    for eval_dataset in "${eval_datasets[@]}" 
    do

        current_time=$(date +"%Y%m%d_%H%M%S")
        OUTPUT_PATH="results/$info/$current_time"
        mkdir -p $OUTPUT_PATH

        python -m torch.distributed.run $DISTRIBUTED_ARGS main_finetune.py \
            --input_size 224 \
            --transform_mode 'crop' \
            --data_path "$train_dataset" \
            --eval_data_path "$eval_dataset" \
            --save_ckpt_freq 2 \
            --batch_size 128 \
            --blr 1e-2 \
            --weight_decay 0.01 \
            --warmup_epochs 1 \
            --epochs 25 \
            --num_workers 8 \
            --output_dir $OUTPUT_PATH \
        2>&1 | tee -a $OUTPUT_PATH/log_train.txt 

    done
done

#        --ratio_train $ratio_data \
#--resume '/home/yiruolei/project/AIGCdetector/SAFE/results/chameleon训练测试/newblock/aa/20250516_171344/checkpoint-10.pth' \
#inputsize 256改成224?会不会太少了,为了和vit相关
#--transform_mode 'crop' \
#--batch_size 32 \
#num_workers 16 \