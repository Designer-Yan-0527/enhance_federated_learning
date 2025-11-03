import torch.nn as nn
from torchvision import models
from config.config import MODEL_CONFIG  # 导入模型配置


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

        # 替换最后的全连接层以适应CIFAR-100的类别数
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, MODEL_CONFIG["num_classes"])

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (tensor): 输入数据，形状为(batch_size, 3, 32, 32)

        返回:
            tensor: 网络输出，形状为(batch_size, 100)，表示每个类别的得分
        """
        return self.resnet(x)
