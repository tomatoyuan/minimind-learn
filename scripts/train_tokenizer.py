import random
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import os

from transformers import AutoTokenizer

random.seed(42)


def train_tokenizer():
    '''从 JSONL 数据集训练 BPE 分词器，定义特殊 Token，保存为标准格式。'''
    # 步骤 1：读取 JSONL 数据集
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']
    
    data_path = './dataset/pretrain_hq.jsonl'

    # 步骤 2：初始化 BPE 分词器
    # BPE（Byte Pair Encoding）：核心思想是「从单字节开始，迭代合并最频繁的字符对」，能平衡词汇量大小和编码效率（既不冗余也不丢失信息）。
    # ByteLevel 预处理：先将文本按 UTF-8 字节拆分（例如 "你好" 拆分为字节 e4 bd a0 e5 a5 bd），再进行 BPE 合并，避免 OOV（未登录词）问题。
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 步骤 3：定义特殊 Token
    # 文本结束符（也用作填充符 pad_token） | 聊天消息开始符 | 聊天消息结束符
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    # 步骤 4：配置 BPE 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=6400, # 训练后词汇表总大小为 6400（含 3 个特殊 Token，实际 BPE 词表为 6397）。
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() # 确保所有单字节都被包含在初始词汇表中，避免字节级 OOV。
    )

    # 步骤 5：训练分词器
    tests = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(tests, trainer=trainer)

    # 步骤 6：设置解码器并验证特殊 Token
    tokenizer.decoder = decoders.ByteLevel # 解码器：与预处理对应，将 BPE token 解码回原始文本

    # 强制验证特殊 Token 的 ID（确保顺序正确，后续聊天模板依赖此 ID）
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # 步骤 7：保存分词器（关键：兼容 Hugging Face 格式）
    #   - tokenizer.json：分词器核心配置（BPE 模型、预处理 / 解码逻辑、词汇表）。
    #   - tokenizer_config.json：transformers 兼容的配置（聊天模板、特殊 Token 映射、模型最大长度等）。
    #       - chat_template：聊天模型的「消息格式化模板」（核心！后续验证会用到），定义了如何将多轮对话（messages）转换为模型输入文本。
    #       - pad_token/eos_token/bos_token：指定填充符、结束符、开始符（对应之前的特殊 Token）。
    #       - model_max_length=32768：模型支持的最大序列长度（可根据实际模型调整）。
    #       - tokenizer_class="PreTrainedTokenizerFast"：指定 transformers 加载时使用的分词器类（确保兼容性）。
    #   - vocab.json + merges.txt：BPE 词汇表和合并规则（由 tokenizer.model.save() 生成）。
    tokenizer_dir = './model/'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("./model/")

    # 手动创建配置文件 tokenizer_config.json
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")

def eval_tokenizer():
    """训练完成后，用 transformers 加载并验证 3 个核心功能："""
    # 步骤 1：加载分词器
    from transformers import PreTrainedTokenizerFast
    tokenizer = AutoTokenizer.from_pretrained("./model/")

    # 步骤 2：验证聊天模板
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False # 直接返回格式化后的字符串，而不是 input_ids
    )
    print(new_prompt)

    # 步骤 3：验证词汇表大小
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('decoder和原始文本是否一致：', response == new_prompt)

def main():
    # train_tokenizer()
    eval_tokenizer()

if __name__ == "__main__":
    main()