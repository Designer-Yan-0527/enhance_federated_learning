import torch
from torch.utils.data import Subset, DataLoader
import random
from data_loader import get_data_loaders, create_task_sequence_datasets
from enhance_federated_learning import EnhancedFederatedLearning
from config import DATA_CONFIG, MODEL_CONFIG
from training_logger import TrainingLogger
import numpy as np


def create_non_iid_clients_for_task(task_dataset, num_clients=5, alpha=0.5):
    """
    为特定任务创建 non-IID 客户端数据划分

    Args:
        task_dataset: 任务数据集
        num_clients: 客户端数量
        alpha: Dirichlet 分布参数，越小 non-IID 程度越高

    Returns:
        客户端数据加载器列表
    """
    # 获取所有数据的标签
    labels = []
    for _, target in task_dataset:
        label = target.item() if isinstance(target, torch.Tensor) else target
        labels.append(label)

    # 获取唯一标签
    unique_labels = list(set(labels))
    num_classes = len(unique_labels)

    # 按类别分组数据索引
    class_indices = [[] for _ in range(num_classes)]
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    for idx, label in enumerate(labels):
        class_indices[label_to_index[label]].append(idx)

    # 使用 Dirichlet 分布划分数据
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        # 对每个类别，使用 Dirichlet 分布确定分配给每个客户端的样本数量
        class_count = len(class_indices[c])
        if class_count == 0:
            continue

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        # 根据比例分配样本
        cumulative_proportions = np.cumsum(proportions)
        shuffled_indices = np.random.permutation(class_indices[c])

        start_idx = 0
        for i in range(num_clients):
            end_idx = int(cumulative_proportions[i] * class_count)
            client_indices[i].extend(shuffled_indices[start_idx:end_idx])
            start_idx = end_idx

    # 收集所有未分配的样本索引
    all_allocated_indices = set()
    for indices in client_indices:
        all_allocated_indices.update(indices)

    all_indices = set(range(len(task_dataset)))
    unallocated_indices = list(all_allocated_indices - all_indices)

    # 创建客户端数据加载器
    client_loaders = []
    for i in range(num_clients):
        # 如果客户端没有样本，从未分配样本中添加或从其他客户端借用
        if len(client_indices[i]) == 0:
            print(f"警告: 客户端 {i} 没有分配到样本，正在补充数据...")
            # 尝试从未分配的数据中添加
            if unallocated_indices:
                # 添加一些未分配的样本
                num_to_add = min(2, len(unallocated_indices))
                client_indices[i].extend(unallocated_indices[:num_to_add])
                unallocated_indices = unallocated_indices[num_to_add:]
            else:
                # 如果没有未分配的数据，从整个数据集中随机选取
                client_indices[i] = np.random.choice(len(task_dataset), 2, replace=False).tolist()

        # 确保客户端至少有2个样本以避免BatchNorm错误
        if len(client_indices[i]) < 2:
            print(f"警告: 客户端 {i} 只有 {len(client_indices[i])} 个样本，正在补充至2个样本")
            while len(client_indices[i]) < 2:
                # 从整个数据集中随机选取补充
                random_idx = np.random.randint(0, len(task_dataset))
                if random_idx not in client_indices[i]:
                    client_indices[i].append(random_idx)

        # 打乱客户端内的数据顺序
        np.random.shuffle(client_indices[i])
        client_dataset = Subset(task_dataset, client_indices[i])
        client_loader = DataLoader(
            client_dataset,
            batch_size=DATA_CONFIG["batch_size"],
            shuffle=True
        )
        client_loaders.append(client_loader)

    return client_loaders


def evaluate_global_model(model, test_loader, task_classes=None):
    """
    在测试集上评估全局模型性能

    Args:
        model: 全局模型
        test_loader: 测试数据加载器
        task_classes: 当前任务的类别（可选）

    Returns:
        包含评估指标的字典
    """
    from config import DEVICE

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            # 如果指定了任务类别，只评估这些类别的准确性
            if task_classes is not None:
                # 过滤出属于当前任务的样本
                mask = torch.zeros_like(target, dtype=torch.bool)
                for cls in task_classes:
                    mask |= (target == cls)

                if mask.sum() == 0:
                    continue

                data = data[mask]
                target = target[mask]

            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    if total == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "correct": 0,
            "total": 0
        }

    avg_loss = total_loss / max(len(test_loader), 1)
    accuracy = 100. * correct / total

    print(f"\n全局模型在测试集上的表现:")
    print(f"  平均损失: {avg_loss:.4f}")
    print(f"  准确率: {accuracy:.2f}% ({correct}/{total})")

    # 返回评估指标以便记录
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def main_continual():
    """支持CIFAR-100的联邦持续学习主函数"""
    print("Starting enhanced continual federated learning with CIFAR-100...")

    # 显示设备信息
    from config import DEVICE
    print(f"Using device: {DEVICE}")

    # 显示关键超参数
    print(f"联邦持续学习配置:")
    print(f"  客户端数量: {MODEL_CONFIG.get('fed_num_clients', 10)}")
    print(f"  联邦轮数: {MODEL_CONFIG.get('fed_num_rounds', 100)}")
    print(f"  本地训练轮数: {MODEL_CONFIG.get('fed_local_epochs', 1)}")
    print(f"  学习率: {MODEL_CONFIG.get('learning_rate', 0.01)}")
    print(f"  批次大小: {DATA_CONFIG.get('batch_size', 64)}")
    print(f"  梯度对齐惩罚系数(gamma): {MODEL_CONFIG.get('fed_gamma', 0.1)}")
    print(f"  EMA权重: {MODEL_CONFIG.get('fed_ema_weight', 0.95)}")
    print(f"  持续学习任务数: {MODEL_CONFIG.get('fcl_num_tasks', 10)}")
    print(f"  每客户端记忆样本数: {MODEL_CONFIG.get('fcl_memory_size', 200)}")

    # 获取原始数据加载器
    train_loader, test_loader = get_data_loaders()

    # 创建任务序列数据集
    task_datasets = create_task_sequence_datasets(train_loader)
    print(f"创建了 {len(task_datasets)} 个任务序列")

    # 创建增强版联邦学习框架实例
    fl_framework = EnhancedFederatedLearning(
        model_type="resnet18",
        ema_weight=MODEL_CONFIG.get("fed_ema_weight", 0.95),
        gamma=MODEL_CONFIG.get("fed_gamma", 0.1)
    )

    # 创建训练日志记录器
    logger = TrainingLogger("continual_logs_cifar100")

    # 按任务序列训练
    num_rounds_per_task = MODEL_CONFIG.get("fed_num_rounds", 100) // len(task_datasets)
    local_epochs = MODEL_CONFIG.get("fed_local_epochs", 1)

    # 首先添加客户端
    num_clients = MODEL_CONFIG.get("fed_num_clients", 10)
    dummy_client_loaders = [train_loader] * num_clients  # 用于初始化客户端
    for i in range(num_clients):
        fl_framework.add_client(i, dummy_client_loaders[i])
        print(f"Added client {i}")

    # 逐个处理任务
    for task_info in task_datasets:
        task_id = task_info['task_id']
        task_dataset = task_info['dataset']
        task_classes = task_info['classes']

        print(f"\n===== 开始学习任务 {task_id + 1}/{len(task_datasets)} =====")
        print(f"任务类别: {task_classes}")

        # 为当前任务创建客户端数据加载器
        use_noniid = MODEL_CONFIG.get("fed_use_noniid", True)
        if use_noniid:
            alpha = MODEL_CONFIG.get("fed_alpha", 0.5)
            print(f"使用 Non-IID 数据划分 (alpha={alpha})")
            client_loaders = create_non_iid_clients_for_task(task_dataset, num_clients, alpha=alpha)
        else:
            print("使用 IID 数据划分")
            # 简单均分数据
            dataset_size = len(task_dataset)
            indices = list(range(dataset_size))
            random.shuffle(indices)
            client_data_size = dataset_size // num_clients

            client_loaders = []
            for i in range(num_clients):
                start_idx = i * client_data_size
                end_idx = start_idx + client_data_size if i < num_clients - 1 else dataset_size
                client_indices = indices[start_idx:end_idx]
                client_dataset = Subset(task_dataset, client_indices)
                client_loader = DataLoader(
                    client_dataset,
                    batch_size=DATA_CONFIG["batch_size"],
                    shuffle=True
                )
                client_loaders.append(client_loader)

        # 注册当前任务
        fl_framework.register_task(task_id, client_loaders, task_classes)

        # 执行当前任务的训练
        for round_num in range(num_rounds_per_task):
            print(f"\n----- 任务 {task_id + 1} 的第 {round_num + 1}/{num_rounds_per_task} 轮 -----")
            # 使用持续学习训练方法
            results = fl_framework.train_continual_round(
                epochs=local_epochs,
                global_lr=MODEL_CONFIG.get("fed_global_lr", 1.0),
                use_replay=True
            )

            # 打印本轮训练结果
            print(f"第 {round_num + 1} 轮训练完成:")
            for result in results["client_results"]:
                print(f"  Client {result['client_id']}: "
                      f"Loss={result['average_loss']:.4f}, "
                      f"Accuracy={result['accuracy']:.2f}%")

            # 在测试集上评估全局模型（仅评估当前任务的类别）
            global_metrics = evaluate_global_model(fl_framework.get_global_model(), test_loader, task_classes)

            # 在计算 noniid_effectiveness 后，将其传递给 logger.log_round
            client_accuracies = [result['accuracy'] for result in results["client_results"]]
            if len(client_accuracies) > 1:
                noniid_effectiveness = np.std(client_accuracies)
            else:
                noniid_effectiveness = 0.0

            # 记录本轮训练日志
            logger.log_round(
                task_id * num_rounds_per_task + round_num + 1,
                global_metrics,
                results["client_results"],
                noniid_effectiveness  # 添加此参数
            )

        # 完成任务，存储记忆
        fl_framework.finish_task()
        print(f"任务 {task_id + 1} 完成，记忆已存储")

        # 评估在所有已见任务上的性能
        all_seen_classes = []
        for i in range(task_id + 1):
            all_seen_classes.extend(task_datasets[i]['classes'])
        cumulative_metrics = evaluate_global_model(fl_framework.get_global_model(), test_loader, all_seen_classes)
        print(f"累积任务准确率 ({len(all_seen_classes)} 个类别): {cumulative_metrics['accuracy']:.2f}%")

    # 保存最终的全局模型
    model_path = "continual_enhanced_federated_resnet18_cifar100.pth"
    fl_framework.save_global_model(model_path)
    print(f"\n持续联邦学习完成，全局模型已保存为 '{model_path}'")

    # 在完整测试集上评估最终全局模型
    final_metrics = evaluate_global_model(fl_framework.get_global_model(), test_loader)

    # 记录最终结果
    logger.save_final_results(final_metrics)
    logger.print_summary()


if __name__ == "__main__":
    main_continual()
