import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.train_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

warnings.filterwarnings('ignore')

# 禁用Flash Attention
import torch.backends.cuda
# 强制使用纯数学计算，虽然慢，但最稳
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

class CriticModel(MiniMindForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 替换lm_head为输出单一价值的线性层
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计
        values = self.value_head(hidden_states).squeeze(-1)
        return values

def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""
    def reasoning_model_reward(rewards):
        # 1. 格式奖励（仅针对训练推理模型时使用）
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 2. 标记奖励（防止严格奖励稀疏，仅针对训练推理模型时使用）
        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)

    # 格式奖励
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 使用reward model计算整个response的奖励
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            scale = 3.0
            score = max(min(score, scale), -scale)

            # 当args.reasoning=1时，额外计算<answer>内容的奖励
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 对answer内容单独计算reward
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, # enc.shape: [Batch_Size, batch_max_seq_Len] 
                       max_length=args.max_seq_len).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]

#Prefix: ======= 插入这段调试代码 =======
        print(f"\n[DEBUG Step]")
        print(f"Input shape: {enc.input_ids.shape}")
        print(f"Max ID in input: {enc.input_ids.max().item()}")
        print(f"Min ID in input: {enc.input_ids.min().item()}")
        print(f"Pad Token ID: {tokenizer.pad_token_id}")
        print(f"EOS Token ID: {tokenizer.eos_token_id}")
        print(f"Model config vocab limit: {actor_model.config.vocab_size}")

        # 严查越界
        if enc.input_ids.max().item() >= actor_model.config.vocab_size:
            print("🔴 CRITICAL ERROR: Input ID exceeds model vocabulary size!")
            print(f"Found ID {enc.input_ids.max().item()} >= {actor_model.config.vocab_size}")
            print("这会导致 CUDA error: device-side assert triggered")
            print("请检查 Tokenizer 是否输出了 6400？如果是，你需要把模型的 vocab_size 设为 6401 或更大。")
            import sys; sys.exit(1)
            
        # 严查 NaN 权重 (防止加载的权重本身就是坏的)
        for name, param in actor_model.named_parameters():
            if torch.isnan(param).any():
                print(f"🔴 CRITICAL ERROR: Parameter {name} contains NaN!")
                import sys; sys.exit(1)

        print(f"Attention Mask Shape: {enc.attention_mask.shape}")
        # 检查每一行 mask 的和
        mask_sums = enc.attention_mask.sum(dim=1)
        print(f"Mask Sums per row: {mask_sums}")
        
        if (mask_sums == 0).any():
            print("🔴 CRITICAL ERROR: Found a row with ALL-ZERO attention mask!")
            print("Reason: One of your prompts is empty or fully filtered out by tokenizer.")
            print("Solution: Check your dataset/jsonl file for empty strings.")
            import sys; sys.exit(1)
            
        # 进一步检查：是不是只有 <bos> 没有其他内容？
        # 如果 mask sum == 1 (只有 bos)，有时候也会导致后续计算不稳定
        if (mask_sums <= 1).any():
            print("⚠️ WARNING: Found a row with extremely short prompt (length <= 1).")
            print("这可能导致 Attention 计算不稳定。")
#Prefix: ==================================
        # torch.full((B,), L): 创建一个长度为 Batch Size (B) 的向量，里面的每个值都是 L。
        # 配合 left padding 使用，表示每个序列的实际内容长度（包含padding），方便后续找出生成内容的起始索引。
        prompt_lengths = torch.full((enc.input_ids.size(0),), enc.input_ids.shape[1], dtype=torch.long, device=enc.input_ids.device)  # [B]

# === 插入测试代码 ===
        print("[Debug] Testing forward pass before generate...")
        with torch.no_grad():
            # 手动跑一次前向传播
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            test_out = model_for_gen(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            test_logits = test_out.logits
            
            if torch.isnan(test_logits).any():
                print("❌ Forward pass produced NaN logits!")
                print(f"Logits max: {test_logits.max()}, min: {test_logits.min()}")
                
                # 进一步检查是哪一层出的问题（如果有Embedding输出NaN，那就是Embedding的问题）
                # 这里假设你有办法访问 embeddings，通常是:
                # print("Embed out:", model_for_gen.model.embed_tokens(enc.input_ids))
                exit(1)
            else:
                print("✅ Forward pass is clean. Logits are finite.")
# ===================

        '''Step 1: 采样 (Rollout)'''
        with torch.no_grad():
            # 如果 actor_model 是 DDP 包装过的，我们需要通过 actor_model.module 访问内部真正的模型，才能调用 .generate()。
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids,          # Prompt 的 Token ID
                attention_mask=enc.attention_mask,# Prompt 的掩码
                max_new_tokens=args.max_gen_len,  # 只限制新生成的 Token 数量 (Response长度)
                do_sample=True,                   # 开启采样 (Sampling)
                temperature=0.8,                  # 温度系数
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        '''Step 2: 计算奖励与价值'''
        responses_text = [tokenizer.decode(gen_out[i, prompt_lengths[i]:], skip_special_tokens=True) for i in range(len(prompts))]
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  # [B]
        
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]

        '''Step 3: 计算优势函数 (Advantage Estimation)'''
        advantages = rewards - values.detach()  # [B]        

        '''Step 4: 计算对数概率 (Log Probabilities)
            1. 当前策略概率：actor_logp
            2. 旧策略概率：old_logp
            3. 参考策略概率：ref_logp
        '''
        # gen_out.shape: [batch_size, seq_len]
        # logits.shape: [batch_size, seq_len, vocab_size]
        logits = actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
        # labels.shape: [batch_size, seq_len - 1]
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        # logits[:, :-1].shape: [batch_size, seq_len - 1, vocab_size] 去掉最后一个时间步的logits，因为没有对应的标签
        # F.log_softmax(logits[:, :-1], dim=-1).shape: [batch_size, seq_len - 1([token_id]), vocab_size(log_prob)]
        # labels.unsqueeze(-1).shape: [batch_size, seq_len - 1, 1(token_id)]
        # F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).shape: [batch_size, seq_len - 1, 1(log_prob)]
        # logp_tokens.shape: [batch_size, seq_len - 1([log_prob])]
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
        seq_len = gen_out.size(1) - 1
        # 将非response部分的log_prob屏蔽掉
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
        # 将非response和padding部分的log_prob屏蔽掉，获得最终的final_mask
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]
        # 对所有生成的有效的log_prob求和，获得最终的actor_logp
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

        with torch.no_grad():
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        '''Step 5: 计算损失函数 (Loss Function)
            PPO的总Loss一般由三部分组成：
            1. 策略损失 (Policy Loss)：让优势（Advantage）高的动作概率变大。
            2. 价值函数损失 (Value Function Loss)：让 Critic 预测得更准。
            3. KL散度惩罚 (KL Divergence Penalty)：强迫 Actor 不要背离 Reference Model (SFT模型) 太远，防止它为了取悦 Reward Model 而输出乱码（Reward Hacking）。
            公式：
            L = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)] + C1 * V_loss + C2 * KL_loss
        '''
        # 1. 策略损失
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        surr1 = ratio * advantages  # [B]
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar
        # 2. 价值函数损失
        value_loss = F.mse_loss(values, rewards)  # scalar
        # 3. KL散度惩罚项
        kl_ref = (actor_logp - ref_logp).mean()
        kl = (actor_logp - old_logp).mean()  # 用于监控：当前策略相对于上一步策略的变化幅度

        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref  # scalar
        loss.backward()


        '''Step 6: 梯度更新与旧策略更新'''
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪在 RLHF 中非常重要，因为强化学习的梯度方差很大，容易导致训练不稳定。
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)  # 梯度裁剪，防止梯度爆炸
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()   # 更新 Actor 参数
            critic_optimizer.step()  # 更新 Critic 参数
            actor_scheduler.step()   # 更新学习率
            critic_scheduler.step()
            actor_optimizer.zero_grad() # 清空梯度
            critic_optimizer.zero_grad()
            torch.cuda.empty_cache() # 稍微清理显存

        if is_main_process():
            # --- 计算生成的平均长度 ---
            response_ids = gen_out[:, enc.input_ids.shape[1]:] # 切分出 Response 部分
            is_eos = (response_ids == tokenizer.eos_token_id)  # 找到 EOS token
            eos_indices = torch.argmax(is_eos.int(), dim=1)    # 找到每行第一个 EOS 的位置
            has_eos = is_eos.any(dim=1)                        # 判断是否有 EOS
            # 如果有 EOS，长度就是 EOS 的索引+1；如果没有，长度就是最大生成长度
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,   # 通常会震荡，不如 Reward 直观。
                    "critic_loss": critic_loss_val, # 通常会震荡，不如 Reward 直观。
                    "reward": reward_val,   # 最重要的指标，应该呈上升趋势。
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,   # 应该维持在一个较低水平，如果飙升说明模型崩了（Mode Collapse）。
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })
            #
            Logger(f"Epoch: {epoch+1}, Step: {step}/{iters}, "
                   f"Actor Loss: {actor_loss_val:.6f}, Critic Loss: {critic_loss_val:.6f}, "
                   f"Reward: {reward_val:.6f}, KL: {kl_val:.6f}, KL_ref: {kl_ref_val:.6f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.2e}, Critic LR: {critic_lr:.2e}")

        if (step + 1) % args.update_old_actor_freq == 0:
            '''
            目的: PPO 要求 ratio 中的分母 π_old 是“采样时的策略”。但为了节省显存和工程方便，这里采用 Rolling Update 策略。
            每隔 update_old_actor_freq 步，就把当前的 actor_model 复制一份给 old_actor_model。
            这样保证了 old_actor 始终紧跟 actor，使得 ratio 接近 1，满足 PPO 的近似条件。
            工程细节: 先转到 CPU 再转回 GPU 或者是为了防止显存碎片化，或者规避某些 DDP 的死锁风险（视具体环境而定）。
            '''
            state_dict = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            actor_state = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            # 保存轻量级权重 (BFloat16/Float16)
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 使用 lm_checkpoint 保存完整状态（包括 critic、优化器状态等，用于断点续训)
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        # --- 激进的显存清理 
        # ---这里显式地 del 掉所有中间变量（特别是计算图相关的 Tensor），并强制清空 CUDA 缓存，是为了防止 OOM (Out of Memory)，确保下一个 Batch 能顺利跑起来。
        del enc, gen_out, responses_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp, old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    parser.add_argument("--reward_model_path", type=str, default="~/models/internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 修正后的逻辑：正确识别 float32
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # 关键修改：如果是 float32，必须禁用 autocast！
    # 否则 autocast 可能会在后台搞鬼
    if dtype == torch.float32:
        autocast_ctx = nullcontext()
        print("DEBUG: Autocast DISABLED for float32 training.")
    else:
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
        print(f"DEBUG: Autocast ENABLED with {dtype}.")
    
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化模型和数据 ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    # Actor模型
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    tokenizer.padding_side = 'left'  # PPO需要左侧padding
    # Old Actor模型
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Critic模型
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    # Reward模型
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    print(f"DEBUG CHECK [Start]")
    print(f"Tokenizer vocab size (len): {len(tokenizer)}")
    print(f"Model vocab size (config): {lm_config.vocab_size}")
    
    # 检查 Embedding 层的实际大小
    if hasattr(actor_model, 'module'):
        # 如果是DDP，取module
        embed_weight = actor_model.module.model.embed_tokens.weight
    else:
        embed_weight = actor_model.model.embed_tokens.weight
        
    print(f"Model Embedding weight shape: {embed_weight.shape}")
    
    if len(tokenizer) > lm_config.vocab_size:
        print("\n[CRITICAL ERROR DETECTED!]")
        print(f"Tokenizer 包含 {len(tokenizer)} 个 token，但模型只定义了 {lm_config.vocab_size} 个槽位。")
        print("这会导致 Embedding 层越界崩溃！")
        exit(1)
    
    print(f"DEBUG CHECK [End]")

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)
    
    # ========== 8. 开始训练 ==========
# 模型权重检查===============================
    print("-" * 30)
    print("正在检查模型权重健康状况...")
    has_nan = False
    for name, param in actor_model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"🔴 CRITICAL WARNING: Parameter [{name}] contains NaN or Inf!")
            print(f"   - Max: {param.max()}, Min: {param.min()}")
            has_nan = True
            
    if has_nan:
        print("❌ 模型权重文件已损坏（包含NaN），无法进行训练。")
        print("请检查 `args.save_dir` 或 `ckp` 指向的路径，删除坏的权重文件，重新开始。")
        exit(1)
    else:
        print("✅ 模型权重检查通过，数值正常。")
    print("-" * 30)
# ==========================================


# ==========================================
# 🕵️‍♂️ NaN 侦探：注册 Hook 精准定位故障层
# ==========================================
    print("\n🕵️‍♂️ 正在注册 NaN 监控钩子 (Layer Hooks)...")
    
    def detect_nan_hook(module, input, output):
        # 1. 提取 Output Tensor
        if isinstance(output, tuple):
            tensor_out = output[0]
        else:
            tensor_out = output
        
        # 2. 检查 NaN 或 Inf
        if isinstance(tensor_out, torch.Tensor):
            if torch.isnan(tensor_out).any() or torch.isinf(tensor_out).any():
                print(f"\n🔴 [CRITICAL ERROR] NaN/Inf DETECTED!")
                print(f"📍 Layer Type: {module.__class__.__name__}")
                print(f"📍 Layer Name: {module}")
                
                # 检查输入情况
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    print(f"   Input Stat: min={input[0].min().item():.4f}, max={input[0].max().item():.4f}, mean={input[0].mean().item():.4f}")
                    if torch.isnan(input[0]).any():
                        print("   (输入本身就已经包含 NaN 了，说明是上一层传下来的)")
                
                # 检查输出情况
                print(f"   Output Stat: {tensor_out}")
                print("🛑 停止运行，请分析上述报错层。")
                import sys; sys.exit(1)

    # 获取实际模型 (解包 DDP)
    real_model = actor_model.module if hasattr(actor_model, "module") else actor_model
    
    # 为每一层注册 Hook
    for name, submodule in real_model.named_modules():
        submodule.register_forward_hook(detect_nan_hook)
        
    print("✅ Hook 注册完成，准备捕捉 NaN...\n")
# ==========================================

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + start_step + 1, old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)