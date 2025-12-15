# minimind-learn

看到项目 minimind 项目，我大为欣喜，基础薄弱我的决定从头复现一遍该项目。看到网上的教程大多是跑通一遍流程，但是我想完全复刻一遍该项目，学习实现细节。

再次感谢作者的伟大贡献，源项目地址：【[minimind](https://github.com/jingyaogong/minimind)】【[minimind-v](https://github.com/jingyaogong/minimind-v)】

以下是个人复现过程中，的一些记录。包含对于项目内容的理解，一些基础知识补充，以及实验结果。

### 项目文档目录

以下按推荐顺序排列，每个章节逐步深入。

| 序号       | 章节标题              | 查看文档                              |
| ---------- | --------------------- | ------------------------------------- |
| 📚 **0**   | 环境搭建              | [查看文档 →](doc/0.环境搭建.md)       |
| 🗂️ **1**   | train tokenizer       | [查看文档 →](doc/1.train_tokenizer.md)|
| 📦 **2**   | DataLoader            | [查看文档 →](doc/2.DataLoader.md)     |
| 🏗️ **3**   | 模型构建              | [查看文档 →](doc/3.模型构建.md)       |
| 🚀 **4**   | Pretrain              | [查看文档 →](doc/4.Pretrain.md)       |
| 🧑‍🏫 **5**   | SFT                   | [查看文档 →](doc/5.SFT.md)            |
| ⚡ **6**   | LoRA                  | [查看文档 →](doc/6.LoRA.md)           |
| 🏆 **7**   | PPO                   | [查看文档 →](doc/7.PPO.md)            |
| ✅ **8**   | DPO                   | [查看文档 →](doc/8.DPO.md)            |
| 🔬 **9**   | 白盒蒸馏              | [查看文档 →](doc/9.白盒蒸馏.md)       |
| 🧩 **10**  | MoE                   | [查看文档 →](doc/10.MoE.md)           |
| 🔄 **11**  | GRPO                  | [查看文档 →](doc/11.GRPO.md)          |
| 👀 **12**  | minimind-v            | [查看文档 →](doc/12.minimind-v.md)    |

---

> 💡 建议按序号从上到下依次阅读，前置章节是后置章节的基础。