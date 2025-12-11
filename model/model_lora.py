import torch
from torch import optim, nn

# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank     # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False) # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # j矩阵B全零初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))  # 前向传播通过A和B矩阵
    

def apply_lora(model, rank=16):
    for name, module in model.named_modules():

        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 如果是 nn.Linear 且为方阵，则插入 LoRA 模块
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)  # 给 module 加一个 lora 成员变量
            original_forward = module.forward  # 保存原始 forward 方法

            # 构造新 forward：原始输出 + LoRA 输出
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora  # 替换 forward 方法

def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if k.startswith(f"{name}.lora.")}
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {f"{name}.lora.{k}": v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)