import json

from torch.utils.data import Dataset # PyTorch 核心库，实现数据集和数据加载；
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false" # 禁用 Tokenizer 的并行处理，避免多线程冲突。

class PretrainDataset(Dataset):
    """
    用于语言模型的无监督预训练（类似 GPT 的自回归预训练），核心任务是 “预测下一个 token”。
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1): # 使用 enumerate 以便在调试时知道行号，行号设置为从1开始
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 构建输入文本
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()  # 去掉批次维度
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  # 创建损失掩码，非填充部分为1，填充部分为0

        X = torch.tensor(input_ids[:-1], dtype=torch.long)  # 输入序列，去掉最后一个token
        Y = torch.tensor(input_ids[1:], dtype=torch.long)   # 目标序列，去掉第一个token
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 损失掩码，去掉第一个token对应的掩码
        return X, Y, loss_mask
    

class SFTDataset(Dataset):
    """
    用于语言模型的有监督指令微调（Supervised Fine-Tuning, SFT），核心任务是 “根据指令和对话历史，生成符合要求的回复”（如聊天机器人、问答系统）。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids # “assistant 回复” 的起始 token ID（如 <s>assistant 对应的 token ID）；
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids # 回复的结束 token ID（如 </s> 对应的 token ID）；

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        # 与 PretrainDataset 的 load_data 逻辑一致，但数据格式不同（需包含 conversations 字段，存储对话历史）。
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    def _create_chat_prompt(self, cs):
        """
        功能：将对话历史（conversations）格式化为模型可识别的文本 prompt；
        示例输出：
            <s>system
            你是一个帮助用户解答数学问题的助手。</s>
            <s>user
            计算 1+1=?</s>
            <s>assistant
            1+1=2</s>
        """
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        # # === 打印每个token的掩码情况 ===
        # print(f"\n--- Sample {index} Token Loss Mask (length: {len(input_ids)}) ---")
        # for i, (token_id, mask) in enumerate(zip(input_ids, loss_mask)):
        #     token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        #     token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')  # 处理换行等不可见字符
        #     print(f"Token {i:3d}: {token_id:5d} -> '{token_str:10s}' | mask: {mask}")
        # print(f"--- End of Sample {index} ---")
        # # ================================
        return X, Y, loss_mask
    

class DPODataset(Dataset):
    pass

class RLAIFDataset(Dataset):
    pass

if __name__ == "__main__":
    pass