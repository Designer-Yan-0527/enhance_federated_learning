import torch
import torch.nn as nn
from torchvision import models
from config import MODEL_CONFIG  # 导入模型配置


class SimpleNN(nn.Module):
    """简单的多层感知机模型（参数从配置文件读取）"""

    def __init__(self):
        """
        初始化神经网络结构
        构建一个三层全连接神经网络：
        输入层(784) -> 隐藏层1(256) -> 隐藏层2(128) -> 输出层(10)
        """
        super(SimpleNN, self).__init__()

        # 使用Sequential容器按顺序组合网络层
        # 从配置文件获取输入/隐藏层/输出层大小
        self.layers = nn.Sequential(
            # 将28x28的图像展平为784维向量
            nn.Flatten(),

            # 第一层全连接层：784维输入 -> 256维隐藏层
            nn.Linear(MODEL_CONFIG["input_size"], MODEL_CONFIG["hidden_size1"]),
            # ReLU激活函数引入非线性
            nn.ReLU(),

            # 第二层全连接层：256维 -> 128维隐藏层
            nn.Linear(MODEL_CONFIG["hidden_size1"], MODEL_CONFIG["hidden_size2"]),
            # ReLU激活函数
            nn.ReLU(),

            # 输出层：128维 -> 10维（对应0-9十个数字类别）
            nn.Linear(MODEL_CONFIG["hidden_size2"], MODEL_CONFIG["output_size"])
        )

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (tensor): 输入数据，形状为(batch_size, 1, 28, 28)

        返回:
            tensor: 网络输出，形状为(batch_size, 10)，表示每个类别的得分
        """
        return self.layers(x)


class ResNet18(nn.Module):
    """使用预训练的ResNet-18作为特征提取器"""

    def __init__(self):
        """
        初始化ResNet-18模型
        """
        super(ResNet18, self).__init__()

        # 使用 weights 参数替代 pretrained 参数
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if MODEL_CONFIG["pretrained"] else None
        self.resnet = models.resnet18(weights=weights)

        # 替换最后的全连接层以适应CIFAR-10的10个类别
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, MODEL_CONFIG["num_classes"])

        # 加载预训练的ResNet-18模型
        self.resnet = models.resnet18(pretrained=MODEL_CONFIG["pretrained"])

        # 替换最后的全连接层以适应CIFAR-10的10个类别
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, MODEL_CONFIG["num_classes"])

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (tensor): 输入数据，形状为(batch_size, 3, 32, 32)

        返回:
            tensor: 网络输出，形状为(batch_size, 10)，表示每个类别的得分
        """
        return self.resnet(x)


# 测试模型（可选）
if __name__ == "__main__":
    model = SimpleNN()
    # 随机生成一个批次的输入（batch_size=32，3通道，32×32）
    dummy_input = torch.randn(32, 3, 32, 32)
    output = model(dummy_input)
    print(f"输入形状：{dummy_input.shape}")
    print(f"输出形状：{output.shape}（应为32×10）")
