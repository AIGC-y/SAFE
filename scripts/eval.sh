
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
RESUME_PATH="/home/yiruolei/project/AIGCdetector/SAFE/results/sdv4test/AIDE双支路优化/newpatch均匀大小不mix其他图片-pixel&lowfre/20250527_094844"
current_time=$(date +"%Y%m%d_%H%M%S")
output_dir=$RESUME_PATH/eval/$current_time
mkdir -p $output_dir


## 只有单纯测试的时候才看每个子目录的情况
eval_datasets=(
    # "data/datasets/test1_ForenSynths/test" \
    # "data/datasets/test2_Self-Synthesis/test" \
    # "/home/yiruolei/ALLDATASET/AIGCDetect/CNNSpot/val"
    # "data/datasets/test3_Ojha/test" \
    # "/home/yiruolei/ALLDATASET/GenImage" \
    "/home/yiruolei/ALLDATASET/AIGCDetect/Chameleon/" \

)

for eval_dataset in "${eval_datasets[@]}"
do
    PYTHONWARNINGS="ignore" python -m torch.distributed.run $DISTRIBUTED_ARGS main_finetune.py \
        --input_size 224 \
        --transform_mode 'ori' \
        --eval_data_path $eval_dataset \
        --batch_size 16  \
        --num_workers 8 \
        --output_dir $output_dir \
        --resume $RESUME_PATH/checkpoint-best.pth \
        --eval True\
        2>&1 | tee -a  $output_dir/log_test.txt 

done
#?lijl
#inputsize 256改成224?会不会太少了,为了和vit相关
#batchsize 256
#checkpoint-best.pth或者last