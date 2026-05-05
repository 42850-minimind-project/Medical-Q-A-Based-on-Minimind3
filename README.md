# Medical Q&A Based on MiniMind3

[English](README.en.md) | 中文

本项目基于 MiniMind3 训练链路，完成了一个面向医疗问答场景的小型语言模型实验。项目重点不是直接使用现成大模型做调用，而是复现并整理从预训练、全量监督微调到医疗 LoRA 适配的完整后训练流程。

> 说明：本项目仅用于学习和实验展示，医疗回答不能替代医生诊断、处方或急救建议。

## 项目目标

- 基于 MiniMind3 代码结构完成小模型训练闭环。
- 使用通用语料进行预训练，获得基础语言续写能力。
- 使用 SFT 数据进行全量监督微调，获得基础指令跟随和对话能力。
- 使用医疗问答 LoRA 数据集进行参数高效微调，观察领域适配效果。
- 对比 `Pretrain`、`Full SFT`、`Full SFT + Medical LoRA` 三个阶段的回答差异。

## 模型结构

本项目使用 MiniMind 的 Dense Transformer 结构作为主要实验模型。模型由 tokenizer、embedding、若干 Transformer layer、RMSNorm、线性输出层和 softmax 组成。Transformer block 内部包含 GQA 注意力和 FFN 模块。

![MiniMind Dense Model](images/LLM-structure.jpg)

MiniMind 也提供 MoE 版本结构，包含 Router 和 Routed Experts。本项目当前训练主线使用 Dense 版本，MoE 图仅作为结构参考。

![MiniMind MoE Model](images/LLM-structure-moe.jpg)

## 训练阶段

### 1. 预训练

预训练阶段使用文本语料训练模型预测下一个 token，主要目标是让模型获得基础语言建模能力。

本项目云端预训练权重命名为：

```text
pretrain_full_4090_seq380_bs32_acc4_768.pth
```

### 2. 全量监督微调

全量 SFT 阶段基于预训练权重继续训练，更新模型全部参数，使模型学习聊天模板、指令跟随和问答格式。

本项目全量 SFT 权重命名为：

```text
full_sft_4090_seq512_bs16_acc8_768.pth
```

### 3. 医疗 LoRA 微调

医疗 LoRA 阶段以 Full SFT 权重作为 base，仅训练 LoRA 增量参数，让模型更适配医疗问答语气、术语和回答结构。

本项目医疗 LoRA 权重命名为：

```text
lora_medical_from_full_sft_768.pth
```

加载医疗 LoRA 推理时需要同时加载：

```text
Full SFT base + Medical LoRA
```

## 数据集

本项目使用的数据集来自 MiniMind 公开数据源和 ModelScope / Hugging Face 镜像。由于数据集和权重文件较大，仓库不直接上传 `.jsonl` 数据和 `.pth` 权重。

本地实验中使用过的数据包括：

- `pretrain_t2t.jsonl`
- `sft_t2t.jsonl`
- `lora_medical.jsonl`
- `dpo.jsonl`
- `rlaif.jsonl`
- `agent_rl.jsonl`
- `agent_rl_math.jsonl`

仓库中的 `.gitignore` 已排除：

```text
dataset/*.jsonl
out/
checkpoints/
```

## 推理示例

只加载预训练权重：

```bash
python eval_llm.py \
  --load_from ./model \
  --weight pretrain_full_4090_seq380_bs32_acc4 \
  --max_new_tokens 256
```

加载全量 SFT 权重：

```bash
python eval_llm.py \
  --load_from ./model \
  --weight full_sft_4090_seq512_bs16_acc8 \
  --max_new_tokens 256
```

加载全量 SFT + 医疗 LoRA：

```bash
python eval_llm.py \
  --load_from ./model \
  --weight full_sft_4090_seq512_bs16_acc8 \
  --lora_weight lora_medical_from_full_sft \
  --max_new_tokens 256
```


### Windows PowerShell 启动方式

如果在本地 Windows PowerShell 中运行，请使用项目自带的 Python 环境，并用反引号 `` ` `` 作为换行符。

只加载预训练权重：

```powershell
.\.conda\python.exe eval_llm.py `
  --load_from .\model `
  --weight pretrain_full_4090_seq380_bs32_acc4 `
  --max_new_tokens 256
```

加载全量 SFT 权重：

```powershell
.\.conda\python.exe eval_llm.py `
  --load_from .\model `
  --weight full_sft_4090_seq512_bs16_acc8 `
  --max_new_tokens 256
```

加载全量 SFT + 医疗 LoRA：

```powershell
.\.conda\python.exe eval_llm.py `
  --load_from .\model `
  --weight full_sft_4090_seq512_bs16_acc8 `
  --lora_weight lora_medical_from_full_sft `
  --max_new_tokens 256
```

## 评估思路

建议使用同一组问题对三个阶段进行对比：

```text
Pretrain
Full SFT
Full SFT + Medical LoRA
```

重点观察：

- 是否理解用户问题。
- 是否能按指令回答。
- 回答是否有结构。
- 医疗问题中是否包含风险提示。
- 是否出现过度诊断、编造或危险建议。

医疗 LoRA 的目标不是让模型替代医生，而是展示小模型在领域语气和回答结构上的适配效果。

## 项目文件

```text
model/                 MiniMind 模型结构与 LoRA 实现
trainer/               预训练、SFT、LoRA、DPO、PPO、GRPO 等训练脚本
dataset/               数据加载代码与数据说明
scripts/               API、Web Demo、模型转换等脚本
eval_llm.py            推理测试脚本
requirements.txt       Python 依赖
```

## 致谢

本项目基于 MiniMind 开源项目进行学习、复现和二次整理：

```text
https://github.com/jingyaogong/minimind
```

感谢原项目提供清晰的小模型训练代码和完整训练流程。
