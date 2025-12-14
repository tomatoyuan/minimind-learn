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
import numpy as np
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
        # values.shape: [batch_size, seq_len]
        return values

class GAE:
    def __init__(self, n_workers: int, worker_steps: int, gamma: float=0.99, lambda_: float=0.95):
        self.n_workers = n_workers
        self.worker_steps = worker_steps
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, done: np.array, rewards: np.array, values: np.array) -> np.array:
        """计算GAE优势函数"""
        # values shape: [B, T], rewards shape: [B, T], done shape: [B, T]
        # 注意：这里的 T 是 max_gen_len
        advantages = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0
        
        # 为了计算方便，通常假设序列结束后的 value 为 0 (或者 bootstrap value)
        # 这里简化处理，假设最后一步之后 value 为 0
        last_value = 0 
        
        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - done[:, t]
            # GAE公式: delta = r + gamma * V(t+1) * (1-done) - V(t)
            # 如果是最后一步，next_value 是 0 (或者 external bootstrap)
            next_val = values[:, t+1] if t + 1 < self.worker_steps else 0
            
            delta = rewards[:, t] + self.gamma * next_val * mask - values[:, t]
            last_advantage = delta + self.gamma * self.lambda_ * mask * last_advantage
            advantages[:, t] = last_advantage
            
        return advantages

def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""
    def reasoning_model_reward(rewards):
        # 1. 格式奖励
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

        # 2. 标记奖励
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

    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

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

            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards

def get_batch_logprobs(model, input_ids, attention_mask, prompt_lengths, max_gen_len):
    """
    计算序列的 Log Probability，并提取 Response 部分对齐到 [B, Max_Gen_Len]
    """
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits  # [B, Seq_Len, V]
    
    # Logits shift: logits[t] 预测 input_ids[t+1]
    logits = logits[:, :-1, :] # [B, Seq_Len-1, V]
    labels = input_ids[:, 1:]  # [B, Seq_Len-1]
    
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather token log probs
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1) # [B, Seq_Len-1]
    
    # 构造对齐的 LogProb 矩阵 [B, Max_Gen_Len]
    batch_size = input_ids.size(0)
    aligned_logprobs = torch.zeros((batch_size, max_gen_len), device=input_ids.device)
    
    for i in range(batch_size):
        p_len = prompt_lengths[i]
        # response 的起始在 labels 中的索引是 p_len - 1
        # (因为 labels 是 input_ids 向左平移一位，所以 input_ids[p_len] 对应 labels[p_len-1])
        start_idx = p_len - 1
        
        # 实际有效长度 (减去 prompt 部分)
        # token_log_probs 的总长度是 seq_len - 1
        curr_resp_len = token_log_probs.size(1) - start_idx
        
        # 截断或取最小值
        safe_len = min(curr_resp_len, max_gen_len)
        
        if safe_len > 0:
            aligned_logprobs[i, :safe_len] = token_log_probs[i, start_idx : start_idx + safe_len]
            
    return aligned_logprobs

def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    actor_model.train()
    critic_model.train()

    # 初始化 GAE 计算器 (worker_steps = max_gen_len)
    gae_solver = GAE(n_workers=loader.batch_size, worker_steps=args.max_gen_len, gamma=0.99, lambda_=0.95)

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, 
                       max_length=args.max_seq_len).to(args.device)
        prompt_lengths = torch.full((enc.input_ids.size(0),), enc.input_ids.shape[1], dtype=torch.long, device=enc.input_ids.device)

        '''Step 1: 采样 (Rollout)'''
        with torch.no_grad():
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        '''Step 2: 计算奖励与价值'''
        responses_text = [tokenizer.decode(gen_out[i, prompt_lengths[i]:], skip_special_tokens=True) for i in range(len(prompts))]
        
        # 2.1 计算整句最终奖励 (Scalar Reward) [B]
        final_rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)
        
        # 2.2 获取整个序列的 Critic Value [B, Seq_Len]
        full_mask = (gen_out != tokenizer.pad_token_id).long()
        # 注意：这里需要 detach 吗？通常 Rollout 阶段不需要梯度，
        # 但如果是为了后面计算 Value Loss 时复用计算图，则需要。
        # 标准 PPO 实现中，Rollout 的 value 是 detach 的，训练时重新计算一次 forward 或者这里保留图。
        # 为了节省显存，通常 detach，训练时再算一次。这里我们先计算出全部 value 用于 GAE。
        with torch.no_grad():
             values_seq_all = critic_model(input_ids=gen_out, attention_mask=full_mask) # [B, Seq_Len]

        # --- 数据对齐与 GAE 准备 ---
        batch_size = len(prompts)
        max_gen_len = args.max_gen_len
        
        # 构造 Dense 矩阵 [B, Max_Gen_Len]
        rewards_dense = torch.zeros((batch_size, max_gen_len), device=args.device)
        values_dense = torch.zeros((batch_size, max_gen_len), device=args.device)
        done_dense = torch.ones((batch_size, max_gen_len), device=args.device) # 默认全done (mask=0)
        valid_mask = torch.zeros((batch_size, max_gen_len), device=args.device) # 用于计算Loss的mask

        for i in range(batch_size):
            p_len = prompt_lengths[i]
            # 计算 response 的实际长度 (遇到 EOS 或 padding 截止)
            # gen_out 包含 prompt，所以要减去 p_len
            total_len = (gen_out[i] != tokenizer.pad_token_id).sum().item()
            r_len = total_len - p_len
            
            if r_len <= 0: continue # 异常保护
            r_len = min(r_len, max_gen_len)

            # 填充 Value (从 prompt 结束后的位置开始取)
            # values_seq_all[i] 对应 input_ids[i]
            # input_ids[p_len] 是第一个生成的 token，它的 value 是 values_seq_all[i, p_len]
            values_dense[i, :r_len] = values_seq_all[i, p_len : p_len + r_len]
            
            # 填充 Reward (稀疏奖励：仅在最后一个有效 token 给分)
            rewards_dense[i, r_len - 1] = final_rewards[i]
            
            # 填充 Done (中间步骤为0，最后一步为1)
            done_dense[i, :r_len-1] = 0.0
            done_dense[i, r_len-1] = 1.0
            
            # Mask (有效部分为1)
            valid_mask[i, :r_len] = 1.0

        # 转为 numpy 供 GAE 使用
        r_np = rewards_dense.cpu().numpy()
        v_np = values_dense.cpu().numpy()
        d_np = done_dense.cpu().numpy()

        '''Step 3: 计算优势函数 (GAE)'''
        # GAE 计算的是 Advantage [B, Max_Gen_Len]
        adv_np = gae_solver(d_np, r_np, v_np)
        advantages = torch.tensor(adv_np, device=args.device, dtype=torch.float32)
        
        # 计算 Returns (用于 Critic 训练的目标值) = Advantage + Value
        returns = advantages + values_dense
        
        # 优势归一化 (只在 valid_mask 范围内)
        # 防止 padding 部分的 0 拉低均值
        if valid_mask.sum() > 0:
            adv_mean = (advantages * valid_mask).sum() / valid_mask.sum()
            adv_std = torch.sqrt(((advantages - adv_mean)**2 * valid_mask).sum() / valid_mask.sum() + 1e-8)
            advantages = (advantages - adv_mean) / adv_std
        
        '''Step 4: 计算对数概率 (Log Probabilities) - Token Level'''
        # 获取 [B, Max_Gen_Len] 的 log probs
        actor_logp = get_batch_logprobs(actor_model, gen_out, full_mask, prompt_lengths, max_gen_len)
        
        with torch.no_grad():
            old_logp = get_batch_logprobs(old_actor_model, gen_out, full_mask, prompt_lengths, max_gen_len)
            ref_logp = get_batch_logprobs(ref_model, gen_out, full_mask, prompt_lengths, max_gen_len)

        '''Step 5: 计算损失函数 (Loss Function)'''
        # 5.1 KL 散度 (Token level)
        # log(p) - log(ref) = log(p/ref)
        kl_per_token = actor_logp - ref_logp
        kl_loss = (kl_per_token * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        # 5.2 策略损失 (PPO Clipped Loss)
        ratio = torch.exp(actor_logp - old_logp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
        
        policy_loss_map = -torch.min(surr1, surr2)
        policy_loss = (policy_loss_map * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        # 5.3 价值函数损失
        # 需要重新计算带梯度的 Value
        # 这里为了节省显存，我们只对 Response 部分做 forward 或者切片
        # 由于我们已经有了 values_dense (detach的)，我们需要重新 forward Critic 来拿梯度
        # 或者更高效的做法：在 Step 2 不 detach，但那样显存开销极大。
        # 这里选择重新 forward 一次 critic
        curr_values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)
        curr_values_dense = torch.zeros_like(values_dense)
        for i in range(batch_size):
            p_len = prompt_lengths[i]
            r_len = int(valid_mask[i].sum().item())
            if r_len > 0:
                curr_values_dense[i, :r_len] = curr_values_seq[i, p_len : p_len + r_len]
        
        value_loss = (F.mse_loss(curr_values_dense, returns.detach(), reduction='none') * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        # 总 Loss
        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_loss
        loss.backward()

        '''Step 6: 梯度更新与旧策略更新'''
        if (step + 1) % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            torch.cuda.empty_cache()

        if is_main_process():
            # 计算平均生成长度
            avg_len = valid_mask.sum(dim=1).float().mean().item()
            
            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = final_rewards.mean().item() # 原始 Reward
            kl_val = ((actor_logp - old_logp) * valid_mask).sum() / (valid_mask.sum() + 1e-8) # Approx KL
            kl_ref_val = kl_loss.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "reward": reward_val,
                    "kl": kl_val.item(),
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len,
                    "actor_lr": actor_lr,
                })
            
            Logger(f"Epoch: {epoch+1}, Step: {step}/{iters}, "
                   f"Actor Loss: {actor_loss_val:.6f}, Critic Loss: {critic_loss_val:.6f}, "
                   f"Reward: {reward_val:.6f}, KL_ref: {kl_ref_val:.6f}, "
                   f"Avg Len: {avg_len:.1f}")

        if (step + 1) % args.update_old_actor_freq == 0:
            state_dict = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            actor_state = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        del enc, gen_out, responses_text, final_rewards, full_mask, values_seq_all
        del rewards_dense, values_dense, done_dense, valid_mask, advantages, returns
        del actor_logp, old_logp, ref_logp, curr_values_seq, curr_values_dense
        del loss, policy_loss, value_loss, kl_loss
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # ... (保持原来的 main 函数不变)
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
    parser.add_argument("--max_gen_len", type=int, default=512, help="生成的最大长度 (减小以节省显存)")
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

    # ... (后续 main 逻辑保持原样，直接复制你的代码即可)
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
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

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
    
    if hasattr(actor_model, 'module'):
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