import torch
from torch.utils.data import Subset, DataLoader
import random
from data_loader import get_data_loaders
from enhance_federated_learning import EnhancedFederatedLearning
from config import DATA_CONFIG, MODEL_CONFIG
from training_logger import TrainingLogger


def create_federated_clients(train_loader, num_clients=5):
    """
    将训练数据分割成多个客户端

    Args:
        train_loader: 原始训练数据加载器
        num_clients: 客户端数量

    Returns:
        客户端数据加载器列表
    """
    # 获取完整的训练数据集
    train_dataset = train_loader.dataset

    # 创建数据索引
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    # 计算每个客户端的数据量
    client_data_size = dataset_size // num_clients

    # 分割数据给各个客户端
    client_loaders = []
    for i in range(num_clients):
        start_idx = i * client_data_size
        end_idx = start_idx + client_data_size if i < num_clients - 1 else dataset_size
        client_indices = indices[start_idx:end_idx]

        # 创建子数据集
        client_dataset = Subset(train_dataset, client_indices)
        client_loader = DataLoader(
            client_dataset,
            batch_size=DATA_CONFIG["batch_size"],
            shuffle=True
        )
        client_loaders.append(client_loader)

    return client_loaders


def evaluate_global_model(model, test_loader):
    """
    在测试集上评估全局模型性能

    Args:
        model: 全局模型
        test_loader: 测试数据加载器

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
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader)
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


def main():
    """主函数"""
    print("Starting enhanced federated learning...")

    # 显示设备信息
    from config import DEVICE
    print(f"Using device: {DEVICE}")

    # 显示关键超参数
    print(f"联邦学习配置:")
    print(f"  客户端数量: {MODEL_CONFIG.get('fed_num_clients', 5)}")
    print(f"  联邦轮数: {MODEL_CONFIG.get('fed_num_rounds', 10)}")
    print(f"  本地训练轮数: {MODEL_CONFIG.get('fed_local_epochs', 2)}")
    print(f"  学习率: {MODEL_CONFIG.get('learning_rate', 0.001)}")
    print(f"  批次大小: {DATA_CONFIG.get('batch_size', 64)}")
    print(f"  梯度对齐惩罚系数(gamma): {0.1}")
    print(f"  EMA权重: {0.95}")

    # 获取原始数据加载器
    train_loader, test_loader = get_data_loaders()

    # 创建增强版联邦学习框架实例
    fl_framework = EnhancedFederatedLearning(
        model_type="resnet18",
        ema_weight=0.95,
        gamma=0.1
    )

    # 创建客户端数据加载器
    num_clients = MODEL_CONFIG.get("fed_num_clients", 5)
    client_loaders = create_federated_clients(train_loader, num_clients)

    # 添加客户端到联邦学习框架
    for i, client_loader in enumerate(client_loaders):
        fl_framework.add_client(i, client_loader)
        print(f"Added client {i} with {len(client_loader.dataset)} samples")

    # 创建训练日志记录器
    logger = TrainingLogger()

    # 进行多轮联邦学习
    num_rounds = MODEL_CONFIG.get("fed_num_rounds", 10)
    local_epochs = MODEL_CONFIG.get("fed_local_epochs", 2)

    for round_num in range(num_rounds):
        print(f"\n===== 开始联邦学习第 {round_num + 1} 轮 =====")
        results = fl_framework.train_round(epochs=local_epochs, global_lr=1.0)

        # 打印本轮训练结果
        print(f"第 {round_num + 1} 轮训练完成:")
        for result in results["client_results"]:
            print(f"  Client {result['client_id']}: "
                  f"Loss={result['average_loss']:.4f}, "
                  f"Accuracy={result['accuracy']:.2f}%")

        # 在测试集上评估全局模型
        global_metrics = evaluate_global_model(fl_framework.get_global_model(), test_loader)

        # 记录本轮训练日志
        logger.log_round(round_num + 1, global_metrics, results["client_results"])

    # 保存最终的全局模型
    fl_framework.save_global_model("enhanced_federated_resnet18.pth")
    print("\n联邦学习完成，全局模型已保存为 'enhanced_federated_resnet18.pth'")

    # 在测试集上评估最终全局模型
    final_metrics = evaluate_global_model(fl_framework.get_global_model(), test_loader)

    # 记录最终结果
    logger.save_final_results(final_metrics)
    logger.print_summary()


if __name__ == "__main__":
    main()
