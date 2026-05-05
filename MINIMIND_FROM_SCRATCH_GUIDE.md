# MiniMind 从零训练与简历项目路线图

本指南基于官方仓库 `jingyaogong/minimind` 当前主线 `minimind-3`。仓库已克隆到：

```text
D:\PythonProjects\minimind
```

你的机器环境初步检查结果：

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8GB 显存
- 已有 Conda: `conda 25.7.0`
- 已有 `torch_env`: PyTorch `2.9.1+cu128`, CUDA 可用
- 但 `torch_env` 位于 `D:\Anaconda\envs`，当前权限不能向该环境安装新包

因此建议新建一个专用环境。RTX 4060 8GB 可以跑通 MiniMind 的轻量训练闭环，但不建议一开始追求完整主线效果。目标应是：先复现 `pretrain_t2t_mini + sft_t2t_mini`，再做一次自己的小型 LoRA 或领域 SFT，让它成为简历里“可解释、可复现、可展示”的项目。

## 1. 项目理解

MiniMind 的核心流程是：

```text
Tokenizer -> Pretrain -> SFT -> 可选 LoRA/KD/DPO/RLAIF/Agentic RL -> Eval -> API/WebUI/部署
```

第一阶段先不要被 RLHF、Agentic RL、Tool Use 全部带跑。你的简历项目主线建议是：

```text
环境搭建 -> 预训练 -> SFT -> 推理测试 -> 训练曲线记录 -> API/WebUI 展示 -> 简历总结
```

等主线跑通后，再把 LoRA 或 Tool Calling 作为增强点。

## 2. 环境搭建

在 Anaconda Prompt 或 PowerShell 中执行：

```powershell
cd "D:\PythonProjects\minimind"
conda create -p "D:\PythonProjects\minimind\.conda" python=3.10 -y
conda activate "D:\PythonProjects\minimind\.conda"
```

如果 PowerShell 提示 `Run 'conda init' before 'conda activate'`，说明当前 PowerShell 没有初始化 Conda。可以先不激活环境，后续命令直接使用项目内 Python：

```powershell
& "D:\PythonProjects\minimind\.conda\python.exe" -m pip --version
```

安装 PyTorch。你的显卡驱动显示 CUDA 12.7，可优先使用 CUDA 12.6 或 12.8 的 PyTorch 轮子；若其中一个失败，换另一个：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

安装项目依赖：

```powershell
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

验证 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

看到 `True` 和你的 RTX 4060 名称，才继续训练。

## 3. 先跑推理，验证环境

先下载官方训练好的 Transformers 格式模型：

```powershell
modelscope download --model gongjy/minimind-3 --local_dir .\minimind-3
```

如果 ModelScope 下载不顺，可以用 HuggingFace：

```powershell
git clone https://huggingface.co/jingyaogong/minimind-3
```

运行 CLI 推理：

```powershell
python eval_llm.py --load_from .\minimind-3 --max_new_tokens 256
```

这一步的意义是确认：Python、依赖、模型加载、GPU 推理都正常。

## 4. 下载训练数据

最小可复现闭环只需要两个文件，放到 `.\dataset`：

```text
pretrain_t2t_mini.jsonl
sft_t2t_mini.jsonl
```

官方数据地址：

- ModelScope: https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
- HuggingFace: https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main

推荐先手动或用浏览器下载这两个文件，确认最终结构是：

```text
minimind
├─ dataset
│  ├─ pretrain_t2t_mini.jsonl
│  └─ sft_t2t_mini.jsonl
```

这两个文件合计约 2.8GB。

## 5. 8GB 显存训练参数

官方默认参数更偏向 24GB 显存环境。你的 8GB 显存建议先用保守参数跑通。

预训练：

```powershell
cd "D:\PythonProjects\minimind\trainer"
python train_pretrain.py --epochs 1 --batch_size 4 --accumulation_steps 16 --max_seq_len 256 --num_workers 2 --log_interval 20 --save_interval 500 --dtype float16
```

输出权重位置：

```text
D:\PythonProjects\minimind\out\pretrain_768.pth
```

SFT：

```powershell
cd "D:\PythonProjects\minimind\trainer"
python train_full_sft.py --epochs 1 --batch_size 2 --accumulation_steps 8 --max_seq_len 512 --num_workers 2 --log_interval 20 --save_interval 500 --dtype float16
```

输出权重位置：

```text
D:\PythonProjects\minimind\out\full_sft_768.pth
```

如果显存溢出，按这个顺序降：

```text
batch_size -> max_seq_len -> num_hidden_layers/hidden_size
```

例如更保守的调试模型：

```powershell
python train_pretrain.py --epochs 1 --hidden_size 512 --num_hidden_layers 8 --batch_size 4 --accumulation_steps 16 --max_seq_len 256 --num_workers 2 --dtype float16
python train_full_sft.py --epochs 1 --hidden_size 512 --num_hidden_layers 8 --batch_size 2 --accumulation_steps 8 --max_seq_len 512 --num_workers 2 --dtype float16
```

此时评估也要带上 `--hidden_size 512`。

## 6. 测试训练结果

回到项目根目录：

```powershell
cd "D:\PythonProjects\minimind"
python eval_llm.py --load_from .\model --weight pretrain --max_new_tokens 256
python eval_llm.py --load_from .\model --weight full_sft --max_new_tokens 256
```

如果训练时用了 `hidden_size 512`：

```powershell
python eval_llm.py --load_from .\model --weight full_sft --hidden_size 512 --max_new_tokens 256
```

## 7. 可视化与部署

Streamlit WebUI：

```powershell
cd "D:\PythonProjects\minimind\scripts"
streamlit run web_demo.py
```

OpenAI-compatible API：

```powershell
cd "D:\PythonProjects\minimind\scripts"
python serve_openai_api.py
```

测试 API：

```powershell
cd "D:\PythonProjects\minimind\scripts"
python chat_api.py
```

## 8. 简历项目怎么做得像自己的

不要只写“复现 MiniMind”。更好的简历表述是：

```text
从零复现 64M 级 MiniMind 语言模型训练链路，完成 tokenizer 使用、Pretrain、SFT、推理评估与 OpenAI-compatible API 部署；针对 8GB 消费级 GPU 调整 batch size、sequence length 与 gradient accumulation，记录训练 loss、显存占用和生成样例，并完成一个面向特定领域的 LoRA/小样本 SFT 扩展实验。
```

建议最终产出：

- 一份 `README`：项目目标、环境、数据、训练命令、结果截图
- 一张训练曲线图：pretrain loss 和 sft loss
- 一组 before/after 问答样例：预训练 vs SFT
- 一个 API 或 WebUI 截图
- 一个自己的增强实验：例如“校园问答助手”“金融术语问答”“医疗科普问答”等
- 一段反思：8GB 显存限制、参数怎么调、loss 如何变化、模型能力有什么边界

## 9. 推荐学习顺序

第一天：

```text
理解 README -> 配环境 -> 跑官方模型推理
```

第二天：

```text
下载 mini 数据 -> 跑 100 到 500 step 预训练 smoke test -> 记录显存和 loss
```

第三天：

```text
完整跑 1 epoch pretrain -> 跑 SFT -> eval_llm 对比
```

第四天：

```text
接 WebUI/API -> 截图 -> 写 README
```

第五天：

```text
做一个自己的 LoRA 或小型领域 SFT -> 加入简历亮点
```

## 10. 常见问题

`CUDA out of memory`：

先降 `batch_size`，再降 `max_seq_len`。SFT 比 pretrain 更吃显存，先用 `batch_size 1 or 2`。

`torch.cuda.is_available()` 是 `False`：

说明 PyTorch 装成 CPU 版，重新安装 CUDA 版 PyTorch。

SFT 找不到 `pretrain_768.pth`：

确认已经先跑完 `train_pretrain.py`，并且文件在 `minimind\out` 目录。

训练中断：

加 `--from_resume 1`：

```powershell
python train_pretrain.py --from_resume 1
python train_full_sft.py --from_resume 1
```

效果不好：

这是小模型正常现象。简历重点不是“效果超过大模型”，而是你完整理解并复现了 LLM 训练链路。
