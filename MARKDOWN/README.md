# Enhanced Federated Continual Learning 使用说明文档

## 项目概述

这是一个增强版联邦持续学习系统，实现了支持梯度对齐惩罚和指数移动平均(EMA)机制的联邦持续学习框架。项目使用 PyTorch 框架构建，基于 ResNet-18 模型，在 CIFAR-100 数据集上进行训练和测试，支持持续学习场景下的知识累积和遗忘防止。

## 依赖包及版本要求

```bash
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.3.0
numpy>=1.21.0
```


## 项目文件结构与功能说明

```
enhance_federated_learning/
├── config/
│   └── config.py              # 配置参数文件，包含所有可调节的超参数
├── data/
│   ├── data_loader.py         # 数据加载器，处理 CIFAR-100 数据集的预处理和加载
│   └── cifar100_classes.py    # CIFAR-100 类别标签定义文件
├── model/
│   ├── resnet18.py            # ResNet-18 模型实现
│   └── client.py              # 客户端实现，处理本地训练、梯度计算和与全局模型的交互
├── federated_learning/
│   └── framework.py           # 联邦学习主框架实现，包含全局模型管理、客户端协调和模型聚合逻辑
├── utils/
│   ├── training_logger.py     # 训练日志记录器，负责记录训练过程中的各项指标
│   └── predict.py             # 模型预测工具，用于对新图片进行分类预测
├── final_model/               # 模型保存目录
├── logs/                      # 日志保存目录
└── enhance_federated_main_continual.py  # 主程序入口，负责初始化训练流程、数据分割和执行联邦持续学习循环
```


## 安装依赖

```bash
pip install torch>=1.9.0 torchvision>=0.10.0 Pillow>=8.3.0 numpy>=1.21.0
```


## 使用方法

### 训练模型

运行联邦持续学习训练:

```bash
python enhance_federated_main_continual.py
```


### 预测图片

使用训练好的模型进行图片预测:

```bash
python utils/predict.py <image_path> <model_path>
```


示例:
```bash
python utils/predict.py test_image.jpg final_model/continual_enhanced_federated_resnet18_cifar100.pth
```


## 超参数详细说明

### 联邦学习相关参数 (位于 `config/config.py` 的 `MODEL_CONFIG` 中)

| 参数名                | 默认值  | 说明                                   |
|--------------------|------|--------------------------------------|
| `fed_num_clients`  | 10   | 联邦学习客户端数量，决定将训练数据分割成多少份              |
| `fed_num_rounds`   | 100  | 联邦学习总轮数，即全局模型更新的次数                   |
| `fed_local_epochs` | 1    | 每个客户端在每轮联邦学习中进行的本地训练轮数               |
| `fed_ema_weight`   | 0.95 | 指数移动平均权重，用于计算全局梯度EMA，值越接近1表示历史梯度影响越大 |
| `fed_gamma`        | 0.1  | 梯度对齐惩罚系数，控制梯度对齐惩罚项的强度                |
| `fed_sample_ratio` | 0.5  | 客户端采样比例，每轮训练时参与训练的客户端占总客户端的比例        |
| `fed_global_lr`    | 1.0  | 全局学习率，用于控制全局模型参数更新的步长                |
| `fed_use_noniid`   | True | 是否使用 Non-IID 数据划分                    |
| `fed_alpha`        | 0.5  | Dirichlet 分布参数，越小 non-IID 程度越高       |

### 数据相关参数 (位于 `config/config.py` 的 `DATA_CONFIG` 中)

| 参数名             | 默认值  | 说明                |
|-----------------|------|-------------------|
| `batch_size`    | 64   | 每个训练批次包含的样本数量     |
| `shuffle_train` | True | 训练集是否在每个epoch打乱顺序 |

### 模型训练参数 (位于 `config/config.py` 的 [MODEL_CONFIG](file://C:\py\Projects\enhance_federated_learning\config\config.py#L13-L40) 中)

| 参数名                                                                                   | 默认值   | 说明                         |
|---------------------------------------------------------------------------------------|-------|----------------------------|
| `num_classes`                                                                         | 100   | 分类任务的类别数（CIFAR-100有100个类别） |
| `pretrained`                                                                          | False | 是否使用ImageNet预训练权重初始化模型     |
| `learning_rate`                                                                       | 0.01  | 本地训练的学习率                   |
| [optimizer](file://C:\py\Projects\enhance_federated_learning\model\client.py#L46-L49) | "SGD" | 优化器类型（可选：Adam、SGD等）        |

### 持续学习相关参数 (位于 `config/config.py` 的 [MODEL_CONFIG](file://C:\py\Projects\enhance_federated_learning\config\config.py#L13-L40) 中)

| 参数名               | 默认值 | 说明                 |
|-------------------|-----|--------------------|
| `fcl_num_tasks`   | 10  | 持续学习任务数（每个任务10个类别） |
| `fcl_memory_size` | 200 | 每个客户端的记忆样本数        |

## 核心组件说明

### EnhancedFederatedLearning 类

主联邦学习框架，负责:
- 初始化全局模型
- 管理客户端
- 执行联邦学习轮次
- 聚合客户端更新
- 支持任务序列管理

### EnhancedFLClient 类

联邦学习客户端，负责:
- 本地模型训练
- 计算分类器梯度
- 实现梯度对齐惩罚
- 生成模型更新
- 支持持续学习机制（记忆存储和回放）

### TrainingLogger 类

训练日志记录器，负责:
- 记录每轮训练结果
- 保存训练历史到JSON和CSV文件
- 输出训练摘要

## 训练流程

1. **初始化**: 创建全局模型和客户端
2. **任务序列处理**:
   - 按顺序处理每个学习任务
   - 为每个任务创建客户端数据划分
3. **联邦持续学习轮次**:
   - 采样客户端子集
   - 计算全局平均梯度并更新EMA
   - 分发全局模型给客户端
   - 客户端本地训练（支持回放数据防止遗忘）
   - 聚合客户端更新
   - 更新全局模型
4. **任务完成**: 存储当前任务的记忆样本
5. **评估**: 在测试集上评估全局模型性能
6. **记录**: 保存训练日志和最终模型

## 输出文件

- `final_model/continual_enhanced_federated_resnet18_cifar100.pth`: 训练完成的全局模型
- `continual_logs_cifar100/training_log.json`: 训练历史记录(JSON格式)
- `continual_logs_cifar100/training_log.csv`: 训练历史记录(CSV格式)
- `continual_logs_cifar100/final_results.json`: 最终评估结果

## 注意事项

1. 确保系统有足够内存来加载数据集和模型
2. 如使用GPU训练，确保已安装CUDA和cuDNN
3. 训练时间取决于数据集大小、模型复杂度和硬件性能
4. 可根据需要调整超参数以获得更好的性能
5. CIFAR-100数据集将被自动下载到`./data`目录下

## 特色功能

### 联邦持续学习支持
- 任务序列管理：将CIFAR-100的100个类别划分为多个学习任务
- 知识保留机制：通过记忆存储和回放防止灾难性遗忘
- 渐进式学习：支持按任务序列学习，模拟真实应用场景

### 增强机制
- 梯度对齐惩罚：通过`fed_gamma`参数控制，提升模型收敛稳定性
- 指数移动平均(EMA)：通过`fed_ema_weight`参数控制，优化全局梯度估计
- Non-IID数据处理：使用Dirichlet分布进行更真实的客户端数据划分

### 灵活配置
- 支持IID和Non-IID数据划分
- 可调节的客户端数量、训练轮数等参数
- 支持预训练模型初始化