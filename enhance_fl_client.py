import torch
import copy
from model import ResNet18, SimpleNN
from config import MODEL_CONFIG, DEVICE
from typing import Dict


class EnhancedFLClient:
    """
    增强版联邦学习客户端实现，支持梯度对齐惩罚
    """

    def __init__(self, client_id: int, data_loader, model_type: str = "resnet18", gamma: float = 0.1):
        """
        初始化客户端

        Args:
            client_id: 客户端ID
            data_loader: 数据加载器
            model_type: 模型类型
            gamma: 梯度对齐惩罚系数
        """
        self.client_id = client_id
        self.data_loader = data_loader
        self.model_type = model_type
        self.gamma = gamma
        self.ema_gradient = None
        self.global_state = None

        # 初始化本地模型
        if model_type == "resnet18":
            self.local_model = ResNet18()
        elif model_type == "simple":
            self.local_model = SimpleNN()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.local_model = self.local_model.to(DEVICE)

        # 设置优化器
        self.optimizer = torch.optim.Adam(
            self.local_model.parameters(),
            lr=MODEL_CONFIG.get("learning_rate", 0.001)
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def update_local_model(self, global_state_dict: Dict[str, torch.Tensor], ema_gradient=None):
        """
        更新本地模型参数

        Args:
            global_state_dict: 全局模型状态字典
            ema_gradient: EMA梯度
        """
        self.local_model.load_state_dict(global_state_dict)
        self.global_state = copy.deepcopy(global_state_dict)
        self.ema_gradient = ema_gradient

    def compute_classifier_gradient(self):
        """
        计算分类器梯度（简化实现）

        Returns:
            分类器梯度字典
        """
        # 这里简单返回全连接层的梯度作为分类器梯度
        try:
            classifier_grad = {}
            if hasattr(self.local_model, 'resnet') and hasattr(self.local_model.resnet, 'fc'):
                # ResNet情况
                for name, param in self.local_model.resnet.fc.named_parameters():
                    if param.grad is not None:
                        classifier_grad[f'resnet.fc.{name}'] = param.grad.clone()
            elif hasattr(self.local_model, 'layers'):
                # SimpleNN情况 - 假设最后一层是分类器
                for name, param in self.local_model.layers[-1].named_parameters():
                    if param.grad is not None:
                        classifier_grad[f'layers.{len(self.local_model.layers) - 1}.{name}'] = param.grad.clone()

            return classifier_grad if classifier_grad else None
        except (AttributeError, KeyError) as e:
            print(f"Error computing classifier gradient: {e}")
            return None

    def compute_gradient_alignment_penalty(self, local_grads):
        """
        计算梯度对齐惩罚项

        Args:
            local_grads: 本地梯度

        Returns:
            惩罚项梯度
        """
        if self.ema_gradient is None:
            return {}

        penalty_grads = {}
        for key in self.ema_gradient.keys():
            if key in local_grads:
                diff = local_grads[key] - self.ema_gradient[key]
                penalty_grads[key] = self.gamma * diff

        return penalty_grads

    def train_local(self, epochs: int = 1) -> Dict[str, float]:
        """
        在本地数据上训练模型

        Args:
            epochs: 本地训练轮数

        Returns:
            训练结果
        """
        self.local_model.train()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)

                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # 如果有EMA梯度，添加梯度对齐惩罚
                if self.ema_gradient is not None:
                    # 获取当前梯度
                    current_grads = {}
                    for name, param in self.local_model.named_parameters():
                        if param.grad is not None:
                            current_grads[name] = param.grad.clone()

                    # 计算惩罚梯度
                    penalty_grads = self.compute_gradient_alignment_penalty(current_grads)

                    # 应用惩罚到梯度
                    for name, param in self.local_model.named_parameters():
                        if name in penalty_grads:
                            param.grad += penalty_grads[name]

                self.optimizer.step()

                total_loss += loss.item()
                total_samples += data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = total_loss / len(self.data_loader)
        accuracy = 100. * correct_predictions / total_samples

        return {
            "client_id": self.client_id,
            "average_loss": avg_loss,
            "accuracy": accuracy,
            "samples": total_samples
        }

    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """
        获取模型更新量（与全局模型的差值）

        Returns:
            模型更新量字典
        """
        update = {}
        current_state = self.local_model.state_dict()

        for key in current_state.keys():
            if self.global_state and key in self.global_state:
                update[key] = current_state[key] - self.global_state[key]
            else:
                update[key] = current_state[key]

        return update
