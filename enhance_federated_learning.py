# enhance_federated_learning.py
import torch
import copy
from collections import OrderedDict
from typing import List, Dict, Any
from model import ResNet18
from config import DEVICE
from enhance_fl_client import EnhancedFLClient


class EnhancedFederatedLearning:
    """
    增强版联邦学习框架实现，支持梯度对齐惩罚和联邦持续学习
    """

    def __init__(self, model_type="resnet18", ema_weight=0.95, gamma=0.1):
        """
        初始化增强版联邦学习框架

        Args:
            model_type: 模型类型 ("resnet18")
            ema_weight: 指数移动平均权重
            gamma: 梯度对齐惩罚系数
        """
        self.model_type = model_type
        self.ema_weight = ema_weight
        self.gamma = gamma
        self.global_model = self._initialize_model()
        self.clients = []
        self.ema_gradient = None
        self.current_task_id = 0
        self.task_sequence = []

    def _initialize_model(self):
        """初始化全局模型"""
        if self.model_type == "resnet18":
            model = ResNet18()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model.to(DEVICE)

    def add_client(self, client_id: int, client_data_loader):
        """
        添加客户端

        Args:
            client_id: 客户端ID
            client_data_loader: 客户端数据加载器
        """
        client = EnhancedFLClient(client_id, client_data_loader, self.model_type, self.gamma)
        self.clients.append(client)

    def register_task(self, task_id: int, task_data_loaders: List, task_classes: List):
        """
        注册新任务

        Args:
            task_id: 任务ID
            task_data_loaders: 该任务的客户端数据加载器列表
            task_classes: 任务包含的类别列表
        """
        self.current_task_id = task_id
        self.task_sequence.append(task_id)

        # 更新客户端数据
        for i, client in enumerate(self.clients):
            if i < len(task_data_loaders):
                client.register_task_data(task_id, task_data_loaders[i], task_classes)

    def finish_task(self):
        """
        完成当前任务，触发记忆存储
        """
        # 让所有客户端存储当前任务的记忆
        for client in self.clients:
            client.store_task_memory()

    @staticmethod
    def compute_global_classifier_gradient(sampled_clients):
        """
        计算采样客户端分类器的全局平均梯度

        Args:
            sampled_clients: 采样的客户端列表

        Returns:
            全局平均梯度
        """
        classifier_gradients = []

        # 计算每个客户端的分类器梯度
        for client in sampled_clients:
            grad = client.compute_classifier_gradient()
            if grad is not None:
                classifier_gradients.append(grad)

        if not classifier_gradients:
            return None

        # 计算平均梯度
        avg_gradient = {}
        for key in classifier_gradients[0].keys():
            grad_stack = torch.stack([grad[key] for grad in classifier_gradients])
            avg_gradient[key] = torch.mean(grad_stack, dim=0)

        return avg_gradient

    def update_ema_gradient(self, current_gradient):
        """
        使用指数移动平均更新全局梯度估计

        Args:
            current_gradient: 当前梯度
        """
        if self.ema_gradient is None:
            self.ema_gradient = copy.deepcopy(current_gradient)
        else:
            for key in current_gradient.keys():
                self.ema_gradient[key] = (
                        self.ema_weight * self.ema_gradient[key] +
                        (1 - self.ema_weight) * current_gradient[key]
                )

    def sample_clients(self, sample_ratio=0.5):
        """
        从所有客户端中随机采样子集

        Args:
            sample_ratio: 采样比例

        Returns:
            采样的客户端列表
        """
        num_sampled = max(1, int(len(self.clients) * sample_ratio))
        import random
        sampled_indices = random.sample(range(len(self.clients)), num_sampled)
        return [self.clients[i] for i in sampled_indices]

    def _perform_federated_round(self, epochs: int, global_lr: float, use_replay: bool = False) -> Dict[str, Any]:
        """
        执行一轮联邦学习训练的通用逻辑

        Args:
            epochs: 本地训练轮数
            global_lr: 全局步长
            use_replay: 是否使用回放数据防止遗忘（仅在持续学习中使用）

        Returns:
            训练结果统计
        """
        # 采样客户端
        sampled_clients = self.sample_clients()
        print(f"采样客户端数量: {len(sampled_clients)}")

        # 计算全局平均梯度并更新EMA
        global_grad = self.compute_global_classifier_gradient(sampled_clients)
        if global_grad is not None:
            self.update_ema_gradient(global_grad)
            print("更新全局梯度EMA完成")

        # 分发全局模型给采样客户端，并传递EMA梯度
        global_state_dict = self.global_model.state_dict()
        for client in sampled_clients:
            client.update_local_model(global_state_dict, self.ema_gradient)

        # 客户端本地训练
        client_updates = []
        client_results = []

        for client in sampled_clients:
            print(f"Training client {client.client_id}...")
            result = client.train_local(epochs, use_replay=use_replay)
            update = client.get_model_update()
            client_updates.append(update)
            client_results.append(result)

        # 聚合模型更新
        if client_updates:
            aggregated_update = self.aggregate_updates(client_updates)

            # 更新全局模型
            global_state = self.global_model.state_dict()
            for key in global_state.keys():
                # 确保类型一致性
                if global_state[key].dtype != aggregated_update[key].dtype:
                    global_state[key] = global_state[key].to(aggregated_update[key].dtype)
                global_state[key] += global_lr * aggregated_update[key]
            self.global_model.load_state_dict(global_state)

        return {
            "client_results": client_results,
            "sampled_clients": len(sampled_clients)
        }

    def train_continual_round(self, epochs: int = 1, global_lr: float = 1.0,
                              use_replay: bool = True) -> Dict[str, Any]:
        """
        执行一轮联邦持续学习训练
        """
        return self._perform_federated_round(epochs, global_lr, use_replay=use_replay)

    @staticmethod
    def aggregate_updates(client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        聚合客户端模型更新

        Args:
            client_updates: 客户端更新列表

        Returns:
            聚合后的更新
        """
        aggregated_update = OrderedDict()

        # 对于每个参数，计算所有客户端的平均更新
        for param_name in client_updates[0].keys():
            update_stack = torch.stack([update[param_name] for update in client_updates])

            # 检查数据类型，如果是整数类型则转换为浮点数
            if not torch.is_floating_point(update_stack) and not torch.is_complex(update_stack):
                update_stack = update_stack.float()

            aggregated_update[param_name] = torch.mean(update_stack, dim=0)

        return aggregated_update

    def get_global_model(self):
        """获取全局模型"""
        return self.global_model

    def save_global_model(self, path: str):
        """保存全局模型"""
        torch.save(self.global_model.state_dict(), path)

    def load_global_model(self, path: str):
        """加载全局模型"""
        self.global_model.load_state_dict(torch.load(path, map_location=DEVICE))
