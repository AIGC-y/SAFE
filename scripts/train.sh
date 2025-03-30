
GPU_NUM=2
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
    "/home/yiruolei/ALLDATASET/CNNSpot/progan_train" \
)
eval_datasets=(
    "/home/yiruolei/ALLDATASET/CNNSpot/progan_val" \
)
#"/home/yiruolei/ALLDATASET/GenImage" \
MODEL="每个分小块都设置trans"
#这里都是有logging信息的所以不需要自己nohup了

for train_dataset in "${train_datasets[@]}" 
do
    for eval_dataset in "${eval_datasets[@]}" 
    do

        current_time=$(date +"%Y%m%d_%H%M%S")
        OUTPUT_PATH="results/$MODEL/$current_time"
        mkdir -p $OUTPUT_PATH

        python -m torch.distributed.run $DISTRIBUTED_ARGS main_finetune.py \
            --input_size 256 \
            --transform_mode 'ori' \
            --model $MODEL \
            --data_path "$train_dataset" \
            --eval_data_path "$eval_dataset" \
            --save_ckpt_freq 1 \
            --batch_size 32 \
            --blr 1e-2 \
            --weight_decay 0.01 \
            --warmup_epochs 1 \
            --epochs 20 \
            --num_workers 16 \
            --output_dir $OUTPUT_PATH \
        2>&1 | tee -a $OUTPUT_PATH/log_train.txt 

    done
done

#--transform_mode 'crop' \