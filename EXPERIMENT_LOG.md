# MiniMind Pretrain Experiment Log

本文件只记录 MiniMind 预训练实验，包括启动命令、参数配置、输出权重和训练结果。

项目路径：

```text
D:\PythonProjects\minimind
```

项目环境：

```text
D:\PythonProjects\minimind\.conda\python.exe
```

硬件：

```text
GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8GB
PyTorch: 2.11.0+cu126
CUDA available: True
```

## EXP001 Formal Pretrain 8GB

日期：2026-04-27

目的：

```text
使用 pretrain_t2t_mini.jsonl 在 RTX 4060 8GB 上正式从 0 预训练。
```

运行方式：

```text
单机单卡普通 Python 启动，不使用 torchrun。
当前只有 1 张 RTX 4060 Laptop GPU，因此不需要 DDP 分布式训练。
```

数据：

```text
D:\PythonProjects\minimind\dataset\pretrain_t2t_mini.jsonl
```

启动命令：

```powershell
cd D:\PythonProjects\minimind\trainer
& ..\.conda\python.exe train_pretrain.py --epochs 1 --batch_size 2 --accumulation_steps 16 --max_seq_len 256 --num_workers 0 --log_interval 20 --save_interval 500 --dtype float16 --save_weight pretrain_8gb_seq256_bs2_acc16
```

参数配置：

| 参数 | 值 | 说明 |
|---|---|---|
| script | train_pretrain.py | 预训练脚本 |
| epochs | 1 | 训练 1 轮 |
| hidden_size | 768 | 默认模型隐藏层维度 |
| num_hidden_layers | 8 | 默认 Transformer 层数 |
| batch_size | 2 | 单步 batch，适配 8GB 显存 |
| accumulation_steps | 16 | 梯度累积步数 |
| effective_batch_size | 32 | batch_size * accumulation_steps |
| max_seq_len | 256 | 最大 token 序列长度，先用保守配置 |
| learning_rate | 5e-4 | train_pretrain.py 默认学习率 |
| grad_clip | 1.0 | train_pretrain.py 默认梯度裁剪 |
| data_path | ../dataset/pretrain_t2t_mini.jsonl | 默认 mini 预训练数据 |
| from_weight | none | 从随机初始化开始训练 |
| dtype | float16 | 混合精度，降低显存占用 |
| num_workers | 0 | Windows 下先用 0，减少 DataLoader 问题 |
| log_interval | 20 | 每 20 step 打印一次日志 |
| save_interval | 500 | 每 500 step 保存一次权重 |
| save_weight | pretrain_8gb_seq256_bs2_acc16 | 输出权重名前缀 |

输出权重：

```text
D:\PythonProjects\minimind\out\pretrain_8gb_seq256_bs2_acc16_768.pth
```

断点续训命令：

```powershell
cd D:\PythonProjects\minimind\trainer
& ..\.conda\python.exe train_pretrain.py --epochs 1 --batch_size 2 --accumulation_steps 16 --max_seq_len 256 --num_workers 0 --log_interval 20 --save_interval 500 --dtype float16 --save_weight pretrain_8gb_seq256_bs2_acc16 --from_resume 1
```

结果记录：

```text
开始时间：
结束时间：
训练耗时：
初始 loss：
最终 loss：
最高显存：
是否 OOM：
是否中断：
备注：
```
