# sft
# python eval_llm.py \
#     --use_moe 0 \
#     --weight full_sft

# lora
# python eval_llm.py \
#     --use_moe 0 \
#     --weight full_sft \
#     --lora_weight lora_identity

# reasoning
python eval_llm.py \
    --use_moe 0 \
    --weight reason
