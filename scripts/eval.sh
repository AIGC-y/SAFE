
#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate SAFE 

#如果要指定一个或者某几个的话
export CUDA_VISIBLE_DEVICES=1

GPU_NUM=1
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12345

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


# RESUME_PATH="./checkpoint"
# RESUME_PATH="/home/data/yrlbp/ALLWEIGHT/SAFE/results/每个分小块都设置trans/20250329_112755/"
RESUME_PATH="/home/yiruolei/ALLOUT/SAFE-results/每个分小块都设置trans/20250329_112755/"
current_time=$(date +"%Y%m%d_%H%M%S")
eval_datasets=(
    # "data/datasets/test1_ForenSynths/test" \
    # "data/datasets/test2_Self-Synthesis/test" \
    # "/home/yiruolei/ALLDATASET/AIGCDetect/CNNSpot/val"
    # "data/datasets/test3_Ojha/test" \
    # "/home/yiruolei/ALLDATASET/GenImage" \
    "/home/yiruolei/ALLDATASET/AIGCDetect/Chameleon/" 

)

for eval_dataset in "${eval_datasets[@]}"
do
    PYTHONWARNINGS="ignore" python -m torch.distributed.run $DISTRIBUTED_ARGS main_finetune.py \
        --input_size 256 \
        --transform_mode 'ori' \
        --eval_data_path $eval_dataset \
        --batch_size 32 \
        --num_workers 16 \
        --output_dir $RESUME_PATH/eval/$current_time \
        --resume $RESUME_PATH/checkpoint-best.pth \
        --eval True
done
#?lijl
#batchsize 256
#checkpoint-best.pth或者last