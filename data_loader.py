from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from config import DATA_CONFIG, MODEL_CONFIG  # 导入数据配置
import torch


def get_data_loaders():
    """
    获取训练集和测试集的数据加载器（参数从配置文件读取）

    返回:
        tuple: (train_loader, test_loader) 训练和测试数据加载器
    """
    # 定义数据预处理步骤
    # CIFAR100
    transform_train = transforms.Compose([
        # 数据增强：随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 数据增强：随机裁剪并填充
        transforms.RandomCrop(32, padding=4),
        # 转换为PyTorch张量
        transforms.ToTensor(),
        # 标准化数据，使用CIFAR-100的标准统计值
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        # 转换为PyTorch张量
        transforms.ToTensor(),
        # 标准化数据，使用CIFAR-100的标准统计值
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # 下载并准备训练数据集
    train_dataset = datasets.CIFAR100(
        root='./data',  # 数据存储路径
        train=True,  # 指定为训练集
        download=True,  # 如果本地没有则下载
        transform=transform_train  # 应用预处理
    )

    # 下载并准备测试数据集
    test_dataset = datasets.CIFAR100(
        root='./data',  # 数据存储路径
        train=False,  # 指定为测试集
        download=True,  # 如果本地没有则下载
        transform=transform_test  # 应用预处理
    )

    # 创建训练数据加载器（从配置文件读取batch_size和shuffle参数）
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=DATA_CONFIG["batch_size"],  # 批次大小
        shuffle=DATA_CONFIG["shuffle_train"]  # 是否打乱数据顺序
    )

    # 创建测试数据加载器
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=DATA_CONFIG["batch_size"],  # 批次大小
        shuffle=False  # 测试集一般不打乱，保持一致性
    )

    return train_data_loader, test_data_loader


def create_task_sequence_datasets(train_data_loader, num_tasks=None):
    """
    将CIFAR-100数据集划分为多个任务序列用于持续学习
    """
    if num_tasks is None:
        num_tasks = MODEL_CONFIG.get("fcl_num_tasks", 10)

    train_dataset = train_data_loader.dataset
    classes_per_task = 100 // num_tasks  # CIFAR-100有100个类别

    # 获取所有数据的标签
    labels = []
    for _, target in train_dataset:
        label = target.item() if isinstance(target, torch.Tensor) else target
        labels.append(label)

    # 按任务划分数据
    task_datasets = []
    for task_id in range(num_tasks):
        # 选择当前任务的类别范围
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task

        # 筛选属于当前任务的数据
        task_indices = []
        for idx, label in enumerate(labels):
            if start_class <= label < end_class:
                task_indices.append(idx)

        # 创建任务子数据集
        task_dataset = Subset(train_dataset, task_indices)
        task_datasets.append({
            'task_id': task_id,
            'dataset': task_dataset,
            'classes': list(range(start_class, end_class))
        })

    return task_datasets
