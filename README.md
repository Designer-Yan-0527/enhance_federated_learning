# Enhanced Federated Learning 使用说明文档

## 项目概述

这是一个增强版联邦学习系统，实现了支持梯度对齐惩罚和指数移动平均(EMA)机制的联邦学习框架。项目使用 PyTorch 框架构建，基于 ResNet-18 模型，在 CIFAR-10 数据集上进行训练和测试。

## 依赖包及版本要求

```bash
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.3.0
```


## 项目文件结构与功能说明

- `enhance_federated_learning.py`: 联邦学习主框架实现，包含全局模型管理、客户端协调和模型聚合逻辑
- `enhance_federated_main.py`: 主程序入口，负责初始化训练流程、数据分割和执行联邦学习循环
- `enhance_fl_client.py`: 客户端实现，处理本地训练、梯度计算和与全局模型的交互
- `model.py`: 模型定义文件，包含 ResNet-18 和简单神经网络模型实现
- `data_loader.py`: 数据加载器，处理 CIFAR-10 数据集的预处理和加载
- `config.py`: 配置参数文件，包含所有可调节的超参数
- `training_logger.py`: 训练日志记录器，负责记录训练过程中的各项指标
- `classes.py`: CIFAR-10 类别标签定义文件
- `predict.py`: 模型预测工具，用于对新图片进行分类预测

## 安装依赖

```bash
pip install torch>=1.9.0 torchvision>=0.10.0 Pillow>=8.3.0
```


## 使用方法

### 训练模型

运行联邦学习训练:

```bash
python enhance_federated_main.py
```


### 预测图片

使用训练好的模型进行图片预测:

```bash
python predict.py <image_path> <model_path>
```


示例:
```bash
python predict.py test_image.jpg enhanced_federated_resnet18.pth
```


## 超参数详细说明

### 联邦学习相关参数 (位于 `config.py` 的 `MODEL_CONFIG` 中)

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `fed_num_clients` | 10 | 联邦学习客户端数量，决定将训练数据分割成多少份 |
| `fed_num_rounds` | 20 | 联邦学习总轮数，即全局模型更新的次数 |
| `fed_local_epochs` | 5 | 每个客户端在每轮联邦学习中进行的本地训练轮数 |
| `fed_ema_weight` | 0.95 | 指数移动平均权重，用于计算全局梯度EMA，值越接近1表示历史梯度影响越大 |
| `fed_gamma` | 0.1 | 梯度对齐惩罚系数，控制梯度对齐惩罚项的强度 |
| `fed_sample_ratio` | 0.5 | 客户端采样比例，每轮训练时参与训练的客户端占总客户端的比例 |
| `fed_global_lr` | 1.0 | 全局学习率，用于控制全局模型参数更新的步长 |

### 数据相关参数 (位于 `config.py` 的 `DATA_CONFIG` 中)

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `batch_size` | 64 | 每个训练批次包含的样本数量 |
| `shuffle_train` | True | 训练集是否在每个epoch打乱顺序 |

### 模型训练参数 (位于 `config.py` 的 `MODEL_CONFIG` 中)

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `num_classes` | 10 | 分类任务的类别数（CIFAR-10有10个类别） |
| `pretrained` | False | 是否使用ImageNet预训练权重初始化模型 |
| `learning_rate` | 0.01 | 本地训练的学习率 |
| `optimizer` | "Adam" | 优化器类型（可选：Adam、SGD等） |

## 核心组件说明

### EnhancedFederatedLearning 类

主联邦学习框架，负责:
- 初始化全局模型
- 管理客户端
- 执行联邦学习轮次
- 聚合客户端更新

### EnhancedFLClient 类

联邦学习客户端，负责:
- 本地模型训练
- 计算分类器梯度
- 实现梯度对齐惩罚
- 生成模型更新

### TrainingLogger 类

训练日志记录器，负责:
- 记录每轮训练结果
- 保存训练历史到JSON和CSV文件
- 输出训练摘要

## 训练流程

1. **初始化**: 创建全局模型和客户端
2. **联邦学习轮次**:
   - 采样客户端子集
   - 计算全局平均梯度并更新EMA
   - 分发全局模型给客户端
   - 客户端本地训练
   - 聚合客户端更新
   - 更新全局模型
3. **评估**: 在测试集上评估全局模型性能
4. **记录**: 保存训练日志和最终模型

## 输出文件

- `enhanced_federated_resnet18.pth`: 训练完成的全局模型
- `logs/training_log.json`: 训练历史记录(JSON格式)
- `logs/training_log.csv`: 训练历史记录(CSV格式)
- `logs/final_results.json`: 最终评估结果

## 注意事项

1. 确保系统有足够内存来加载数据集和模型
2. 如使用GPU训练，确保已安装CUDA和cuDNN
3. 训练时间取决于数据集大小、模型复杂度和硬件性能
4. 可根据需要调整超参数以获得更好的性能