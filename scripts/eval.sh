
#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate SAFE 



GPU_NUM=2
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

MODEL="SAFE"
RESUME_PATH="./checkpoint"
# RESUME_PATH="./results/SAFE/20250318_114617"

eval_datasets=(
    # "data/datasets/test1_ForenSynths/test" \
    # "data/datasets/test2_Self-Synthesis/test" \
    # "data/datasets/test3_Ojha/test" \
    # "/home/yiruolei/ALLDATASET/GenImage" \
    "/home/yiruolei/ALLDATASET/Chameleon" 

)
for eval_dataset in "${eval_datasets[@]}"
do
    PYTHONWARNINGS="ignore" python -m torch.distributed.run $DISTRIBUTED_ARGS main_finetune.py \
        --input_size 256 \
        --transform_mode 'crop' \
        --model $MODEL \
        --eval_data_path $eval_dataset \
        --batch_size 256 \
        --num_workers 16 \
        --output_dir $RESUME_PATH \
        --resume $RESUME_PATH/checkpoint-best.pth \
        --eval True
done

#batchsize 256
#checkpoint-best.pth或者last