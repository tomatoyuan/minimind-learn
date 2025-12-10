# python train_pretrain.py \
#     --use_moe 0 \
#     --num_hidden_layers 8 \
#     --epochs 6 \
#     --max_seq_len 512 \
#     --data_path /home/qyfan/tomato/src_learning/minimind-learn/dataset/pretrain_hq.jsonl \
#     --from_weight none \
#     --from_resume 1 \
#     --use_wandb \
#     --wandb_project MiniMind-Pretrain


# python train_full_sft.py \
#     --use_moe 0 \
#     --num_hidden_layers 8 \
#     --epochs 20 \
#     --max_seq_len 512 \
#     --data_path /home/qyfan/tomato/src_learning/minimind-learn/dataset/sft_512.jsonl \
#     --from_weight pretrain \
#     --from_resume 0 \
#     --use_wandb \
#     --wandb_project MiniMind-Full-SFT-512


# 单机 4 张卡
torchrun --nproc_per_node=4 --master_port=29511 train_full_sft.py \
    --use_moe 0 \
    --num_hidden_layers 8 \
    --epochs 20 \
    --max_seq_len 512 \
    --data_path /home/qyfan/tomato/src_learning/minimind-learn/dataset/sft_512.jsonl \
    --from_weight pretrain \
    --from_resume 0 \
    --use_wandb \
    --wandb_project MiniMind-Full-SFT-512