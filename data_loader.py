from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import DATA_CONFIG  # 导入数据配置


def get_data_loaders():
    """
    获取训练集和测试集的数据加载器（参数从配置文件读取）

    返回:
        tuple: (train_loader, test_loader) 训练和测试数据加载器
    """
    # 定义数据预处理步骤
    """
    # MNIST
    transform = transforms.Compose([
        # 转换为PyTorch张量
        transforms.ToTensor(),
        # 标准化数据，均值0.1307，标准差0.3081是MNIST数据集的标准统计值
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    """
    # CIFAR10
    transform_train = transforms.Compose([
        # 数据增强：随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 数据增强：随机裁剪并填充
        transforms.RandomCrop(32, padding=4),
        # 转换为PyTorch张量
        transforms.ToTensor(),
        # 标准化数据，使用CIFAR-10的标准统计值
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        # 转换为PyTorch张量
        transforms.ToTensor(),
        # 标准化数据，使用CIFAR-10的标准统计值
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 下载并准备训练数据集
    train_dataset = datasets.CIFAR10(
        root='./data',  # 数据存储路径
        train=True,  # 指定为训练集
        download=True,  # 如果本地没有则下载
        transform=transform_train  # 应用预处理
    )

    # 下载并准备测试数据集
    test_dataset = datasets.CIFAR10(
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


# 测试数据加载（可选）
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    print(f"训练集批次数量：{len(train_loader)}")
    print(f"测试集批次数量：{len(test_loader)}")
