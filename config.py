import torch

# 设备配置（自动选择GPU或CPU）
# 如果系统有可用的CUDA设备，则使用GPU；否则使用CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 数据加载参数
DATA_CONFIG = {
    "batch_size": 64,  # 每个训练批次包含的样本数量
    "shuffle_train": True,  # 训练集是否在每个epoch打乱顺序
}

# 模型参数（如果有多种模型，可在这里区分）
MODEL_CONFIG = {
    # 联邦学习配置
    "fed_num_clients": 10,  # 联邦学习客户端数量
    "fed_num_rounds": 100,  # 联邦学习轮数
    "fed_local_epochs": 1,  # 每轮本地训练轮数

    # Non-IID 配置
    "fed_use_noniid": True,  # 是否使用 Non-IID 数据划分
    "fed_alpha": 0.5,        # Dirichlet 分布参数，越小 non-IID 程度越高

    # 增强版联邦学习配置
    "fed_ema_weight": 0.95,        # 指数移动平均权重
    "fed_gamma": 0.1,              # 梯度对齐惩罚系数
    "fed_sample_ratio": 0.5,       # 客户端采样比例
    "fed_global_lr": 1.0,          # 全局学习率

    # CIFAR10
    "num_classes": 10,      # CIFAR-10有10个类别
    "pretrained": False,    # 是否使用预训练权重
    "epochs": 10,           # 训练轮次（完整遍历训练集的次数）
    "learning_rate": 0.01,  # 学习率（控制参数更新步长）
    "optimizer": "SGD",    # 优化器类型（可选：Adam、SGD等）
    "loss_function": "CrossEntropyLoss",  # 损失函数类型（适用于分类任务）

    # # CIFAR100
    # "input_size": 3072,   # CIFAR100: 32*32*3 = 3072
    # "hidden_size1": 512,  # 增加隐藏层大小以处理更复杂的数据
    # "hidden_size2": 256,
    # "output_size": 100,   # CIFAR100有100个类别
}

# 训练参数
TRAIN_CONFIG = {
    "epochs": 5,          # 训练轮次（完整遍历训练集的次数）
    "learning_rate": 0.01,  # 学习率（控制参数更新步长）
    "optimizer": "SGD",  # 优化器类型（可选：Adam、SGD等）
    "loss_function": "CrossEntropyLoss",  # 损失函数类型（适用于分类任务）
}

# 保存模型参数
SAVE_CONFIG = {
    "save_model": True,   # 是否保存训练好的模型
    "model_path": "cifar10_resnet18.pth"  # 模型保存路径和文件名
}

"""
注意：
1. 配置文件保存在config.py中，可自行添加参数。
2. 配置文件在main.py中导入，并使用。
3. 配置文件在data_loader.py中导入，并使用。
4. 配置文件在model.py中导入，并使用。
5. 配置文件在train_test.py中导入，并使用。
"""
"""超参数配置文件：所有可调节参数集中在这里"""
