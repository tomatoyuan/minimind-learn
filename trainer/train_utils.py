"""
训练工具函数集合
"""
from logging import Logger
import os
import math
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """
    分布式训练中，仅主进程打印日志，避免多进程重复打印。
    """
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    '''余弦退火学习率调度'''
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def init_distributed_mode():
    """
    初始化分布式训练环境。
    如果未启用分布式训练，则不进行任何操作。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非分布式训练，返回 local_rank 为 0
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """
    设置随机种子以确保实验的可重复性。
    """
    # 设置 Python 内置随机数生成器的种子
    random.seed(seed)
    # 设置 NumPy 随机数生成器的种子
    np.random.seed(seed)
    # 固定当前 GPU 的随机种子
    # PyTorch 中 当前使用的单块 GPU 的随机操作（比如 GPU 上的张量随机初始化、GPU 上的 Dropout 等）会使用这个种子；
    torch.cuda.manual_seed(seed)
    # 固定所有 GPU 的随机种子
    # 分布式训练（比如 DDP 模式）中，多块 GPU 同时工作，需要固定所有 GPU 的种子，避免不同 GPU 上的随机操作不一致。
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    # 强制 cuDNN 使用确定性算法
    # 背景：cuDNN 是 NVIDIA 提供的 GPU 加速库，PyTorch 的卷积、池化等操作会依赖它；
    # 作用：默认情况下，cuDNN 为了追求速度，可能会选择非确定性算法（同一输入可能得到不同输出）；设置为 True 后，cuDNN 会强制使用 确定性算法（同一输入一定得到相同输出）；
    torch.backends.cudnn.deterministic = True
    # 关闭 cuDNN 的自动优化（benchmark）
    # benchmark=True 时，cuDNN 会在训练开始前，对当前任务的卷积操作进行 “测速”，选择最快的卷积算法（不同次运行可能选不同算法）；
    # 作用：设置为 False 后，cuDNN 会禁用这种自动选择，每次都使用固定的算法（配合 deterministic=True 进一步保证结果一致）；
    # 补充：如果你的输入数据的形状（比如图片尺寸、batch size）是固定的，benchmark=True 能加速训练，但会牺牲可复现性；实验阶段建议关闭，正式训练时可根据需求开启。
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        # 关键：区分是否为分布式训练模型，正确提取权重
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        ckp_tmp = ckp_path + '.tmp'  # 临时保存路径
        # 保存权重（转半精度float16，减小文件体积）
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        # 原子操作替换临时文件为最终模型文件（避免保存中断导致文件损坏）
        os.replace(ckp_tmp, ckp_path)
        # wandb 是深度学习中常用的实验跟踪工具（记录损失、指标、超参数等）
        # 每个实验会分配一个唯一 id，后续可通过这个 id 复现实验、继续训练或查看历史记录。
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        '''解释 world_size:
        在分布式训练中，world_size 是核心概念，指的是 参与训练的「总进程数」
        （对应总 GPU 数，因为分布式训练中通常「1 个进程绑定 1 块 GPU」）—— 它决定了训练的并行规模，是分布式通信、数据划分的关键参数。
            1. 数据划分校验：分布式训练中，训练数据会按 world_size 分片（每个进程处理 1/world_size 的数据），恢复时需确保当前 world_size 与之前一致，否则数据分片会出错；
            2. 进程通信配置：恢复分布式进程组时，需基于 world_size 确认总进程数，避免通信异常；
            3. 优化器状态恢复：部分优化器（如 AdamW）的状态（动量、方差）可能与并行规模相关，world_size 用于校验恢复环境的一致性。
        '''
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    
    else:  # 加载模式
        if os.path.exists(resume_path):
            # 不管保存时 tensor 在哪个设备（GPU 0、GPU 1、多卡分布式），加载后所有 tensor 都放到 CPU 上；
            # 先加载到 CPU，后续可按需将 tensor 移到 GPU（或分批次移动），避免显存峰值过高。
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                '''
                step：训练步数，指「模型更新参数的次数」（每处理 1 个 batch 数据，模型更新 1 次参数，step 加 1）；
                分布式训练的 batch 逻辑：假设单卡 batch size=8，若用 world_size=2（2 块 GPU），则全局 batch size=8×2=16（每步 2 块 GPU 各处理 8 个样本，联合更新参数）；
                关键结论：相同 step 下，GPU 数量越多，模型处理的总样本数越多（总样本数 = step × 单卡 batch size × world_size）。
                举个例子：
                    之前用 2 块 GPU（saved_ws=2）训练，step=1000 → 总样本数 = 1000 × 8 × 2 = 16000；
                    现在改用 1 块 GPU（current_ws=1）恢复训练，若直接用 step=1000 → 总样本数 = 1000 ×8 ×1=8000（比之前少了一半，进度不一致）；
                    解决方案：将 step 调整为 1000×2÷1=2000 → 新 step 下总样本数 = 2000×8×1=16000（与之前一致）。
                '''
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None

def init_model(lm_config, from_weight='none', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "model") # 避免 debug 时搞错根目录，找不到tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    Logger(f'所加载Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model.to(device), tokenizer

class SkipBatchSampler(Sampler):
    """
    用于跳过指定数量的批次（batches）的采样器包装器。
    适用于在训练过程中需要从特定批次开始继续训练的场景。
    """
    def __init__(self, sampler: Sampler, batch_size: int, skip_batches: int = 0):
        self.sampler = sampler          # 底层的原始 sampler（比如 SequentialSampler 或 RandomSampler）
        self.batch_size = batch_size    # 每个 batch 的大小
        self.skip_batches = skip_batches  # 要跳过（丢弃）多少个完整的 batch
    
    def __iter__(self):
        '''
        跳过（丢弃）数据集中最前面的 skip_batches 个 batch，其余的 batch 正常返回。
        '''
        batch = []              # 临时存放当前正在构造的 batch
        skipped = 0             # 记录已经跳过了多少个 batch

        for idx in self.sampler:        # 遍历底层 sampler 的每一个 index
            batch.append(idx)
            if len(batch) == self.batch_size:   # 一个 batch 凑齐了
                if skipped < self.skip_batches: # 还没跳够
                    skipped += 1                 # 跳过计数 +1
                    batch = []                   # 扔掉这个 batch，重新开始
                    continue                     # 进入下一个
                yield batch          # 跳够了！开始正常输出这个 batch
                batch = []           # 清空，准备下一个 batch

        # 处理最后可能剩下的不完整 batch
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch              # 如果已经跳够了，就把剩下的也 yield（可选行为）

    def __len__(self):
        # 凑不够一个batch_size的按照一个batch计算
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)