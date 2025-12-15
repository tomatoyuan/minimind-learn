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
#     --epochs 2 \
#     --max_seq_len 512 \
#     --data_path /home/qyfan/tomato/src_learning/minimind-learn/dataset/sft_512.jsonl \
#     --from_weight pretrain \
#     --from_resume 1 \
#     --use_wandb \
#     --wandb_project MiniMind-Full-SFT-512


# 单机 4 张卡
# torchrun --nproc_per_node=4 --master_port=29511 train_full_sft.py \
#     --use_moe 0 \
#     --num_hidden_layers 8 \
#     --epochs 20 \
#     --max_seq_len 512 \
#     --data_path /home/qyfan/tomato/src_learning/minimind-learn/dataset/sft_512.jsonl \
#     --from_weight pretrain \
#     --from_resume 0 \
#     --use_wandb \
#     --wandb_project MiniMind-Full-SFT-512


# python train_lora.py \
#     --use_moe 0 \
#     --num_hidden_layers 8 \
#     --epochs 50 \
#     --max_seq_len 512 \
#     --data_path /home/qyfan/tomato/src_learning/minimind-learn/dataset/lora_identity.jsonl \
#     --from_weight full_sft \
#     --from_resume 0 \
#     --use_wandb \
#     --wandb_project MiniMind-LoRA


# python train_ppo.py \
#     --epochs 3 \
#     --use_moe 0 \
#     --batch_size 16 \
#     --learning_rate 1e-6 \
#     --dtype bfloat16 \
#     --data_path /home/qyfan/tomato/src_learning/minimind-learn/dataset/rlaif-mini.jsonl \
#     --reasoning 0 \
#     --update_old_actor_freq 4 \
#     --reward_model_path /home/qyfan/models/internlm2-1_8b-reward \
#     --from_resume 0 \
#     --use_wandb \
#     --wandb_project MiniMind-PPO

# python train_dpo.py \
#     --epochs 2 \
#     --use_moe 0 \
#     --batch_size 16 \
#     --use_wandb \
#     --wandb_project MiniMind-DPO

# python train_distill_reason.py \
#     --epochs 1 \
#     --use_wandb \
#     --wandb_project MiniMind-Distill-Reason

# MoE
# python train_pretrain.py \
#     --epochs 1 \
#     --use_moe 1 \
#     --use_wandb \
#     --wandb_project MiniMind-Pretrain-MOE

# python train_grpo.py \
#     --reward_model_path /home/qyfan/models/internlm2-1_8b-reward \
#     --use_wandb \
#     --wandb_project MiniMind-GRPO


### vlm ###
# python train_pretrain_vlm.py \
#     --from_weight llm \
#     --use_wandb \
#     --wandb_project MiniMind-V-Pretrain


python train_sft_vlm.py \
    --from_weight pretrain_vlm \
    --use_wandb \
    --wandb_project MiniMind-V-SFT