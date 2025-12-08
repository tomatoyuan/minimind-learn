import os
import sys

# 声明当前脚本属于 trainer 包，规范包结构；允许 import trainer.xxx 的导入方式
__package__ = "trainer"
# 把项目根目录加入 Python 模块搜索路径，让脚本能导入上级目录的资源。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.train_utils import get_lr, init_distributed_mode, setup_seed, lm_checkpoint, is_main_process, init_model, SkipBatchSampler, Logger

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练一个 epoch 的完整函数
    参数:
        epoch: 当前是第几个 epoch（从 0 开始）
        loader: DataLoader（已经通过 SkipBatchSampler 跳过了已训练数据）
        iters: 本 epoch 总共多少个 step（= len(loader)）
        start_step: 从第几个 global step 开始算（用于续训时日志对齐）
        wandb: 是否启用 wandb 日志
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none') # 不自动求 mean，方便后面用 loss_mask 加权
    start_time = time.time()                         # 记录本 epoch 开始时间，用于计算 ETA

    # enumerate(loader, start=start_step + 1) 
    # → 从 start_step+1 开始编号，让全局 step 是连续的（续训时日志不会从 1 开始）
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # ==================== 动态设置当前学习率 ====================
        # 全局 step = epoch * iters_per_epoch + 当前 step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 把学习率应用到优化器的所有参数组（支持不同层不同 lr）
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ==================== 混合精度前向 ====================
        with autocast_ctx:  # 自动使用 fp16/bf16，极大省显存、提速
            res = model(X)  # 前向，返回包含 logits 和 aux_loss 的命名元组
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # 只计算有效 token 的 loss（padding 和 prompt 部分通常 mask 掉）
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 加上 MoE 路由的辅助损失（平衡专家负载）
            loss += res.aux_loss
            # 梯度累积：真实 loss 除以累积步数
            loss = loss / args.accumulation_steps

        # ==================== 反向传播 ====================
        scaler.scale(loss).backward()   # 放大 loss → 防止 fp16 下溢

        # ==================== 梯度累积到够一步才更新 ====================
        if (step + 1) % args.accumulation_steps == 0:
            # 1. unscale 梯度（准备剪裁）
            scaler.unscale_(optimizer)
            # 2. 梯度裁剪，防止梯度爆炸（大模型训练必备！）
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 3. 优化器一步 + scaler 更新缩放因子
            scaler.step(optimizer)
            scaler.update()

            # 4. 清零梯度（set_to_none=True 更快省显存）
            optimizer.zero_grad(set_to_none=True)
            # 5. 清理显存碎片（可选，但推荐）
            torch.cuda.empty_cache()

        # ==================== 日志打印 & wandb ====================
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算本 epoch 还剩多少分钟
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # ==================== 保存检查点 ====================
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval() # 临时切 eval，避免 BN/Dropout 影响保存
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # DDP 模型要保存 module.state_dict()
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            # 自定义的完整 checkpoint（包含 config、optimizer、scaler、step 等）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    # zero 是指：跑一遍进行流程验证、快速试错、初步效果评估
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="/home/qyfan/tomato/src_learning/minimind-learn/dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=1, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练模式，获取当前进程的「本地 GPU 编号（local_rank）」。如果是「单机单卡训练」（非分布式），这个函数通常会返回 0，且不初始化进程组。
    local_rank = init_distributed_mode()
    # 如果启用分布式训练，每个进程绑定到对应的 GPU 上。
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 固定随机种子，同时保证分布式训练中「各进程的种子不同但固定」（避免多进程重复计算）。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 单卡训练时数据集需要shuffle，分布式训练时交给 DistributedSampler 处理 shuffle
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # PyTorch 提供的梯度缩放器（Gradient Scaler），主要用于混合精度训练（AMP）
    # scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data.get('epoch', 0)
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点，首次进入续训，跳过已完成的 step
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step * args.accumulation_steps)
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb)