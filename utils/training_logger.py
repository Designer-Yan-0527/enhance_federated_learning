import json
import csv
import os
from typing import List, Dict, Any
import time


class TrainingLogger:
    """
    训练日志记录器，用于记录联邦学习过程中每一轮的loss和准确率
    """

    def __init__(self, log_dir: str = "logs"):
        """
        初始化训练日志记录器

        Args:
            log_dir: 日志文件存储目录
        """
        self.log_dir = log_dir
        self.training_history = []
        self.start_time = time.time()  # 记录开始时间

        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 日志文件路径
        self.json_log_file = os.path.join(log_dir, "training_log.json")
        self.csv_log_file = os.path.join(log_dir, "training_log.csv")

        # 初始化CSV文件头
        self._initialize_csv()

    def _initialize_csv(self):
        """初始化CSV文件并写入表头"""
        if not os.path.exists(self.csv_log_file):
            with open(self.csv_log_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'round',
                    'global_loss',
                    'global_accuracy',
                    'elapsed_time',
                    'noniid_effectiveness',
                    'client_details'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    def log_round(self, round_num: int, global_eval_metrics: Dict[str, float],
                  clients_data: List[Dict[str, Any]], noniid_effectiveness: float = 0.0):
        """
        记录一轮训练的结果

        Args:
            round_num: 联邦学习轮次
            global_eval_metrics: 全局模型在测试集上的评估指标
            clients_data: 各客户端的训练结果
            noniid_effectiveness: NonIID识别效果指标
        """
        elapsed_time = time.time() - self.start_time

        # 构造日志条目
        log_entry = {
            "round": round_num,
            "timestamp": self._get_timestamp(),
            "global_metrics": global_eval_metrics,
            "client_results": clients_data,
            "elapsed_time": elapsed_time,
            "noniid_effectiveness": noniid_effectiveness
        }

        # 添加到历史记录
        self.training_history.append(log_entry)

        # 保存到JSON文件
        self._save_to_json()

        # 保存到CSV文件
        self._save_to_csv(round_num, global_eval_metrics, clients_data, elapsed_time, noniid_effectiveness)

        print(f"Round {round_num} metrics logged successfully.")

    @staticmethod
    def _get_timestamp() -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _save_to_json(self):
        """保存日志到JSON文件"""
        with open(self.json_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)

    def _save_to_csv(self, round_num: int, global_eval_metrics: Dict[str, float],
                     clients_data: List[Dict[str, Any]], elapsed_time: float,
                     noniid_effectiveness: float):
        """保存关键指标到CSV文件"""
        with open(self.csv_log_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                'round',
                'global_loss',
                'global_accuracy',
                'elapsed_time',
                'noniid_effectiveness',
                'client_details'
            ])
            writer.writerow({
                'round': round_num,
                'global_loss': global_eval_metrics.get('loss', ''),
                'global_accuracy': global_eval_metrics.get('accuracy', ''),
                'elapsed_time': elapsed_time,
                'noniid_effectiveness': noniid_effectiveness,
                'client_details': json.dumps(clients_data, ensure_ascii=False)
            })

    def save_final_results(self, final_metrics: Dict[str, float]):
        """
        保存最终结果

        Args:
            final_metrics: 最终评估指标
        """
        final_log_file = os.path.join(self.log_dir, "final_results.json")
        with open(final_log_file, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    def load_history(self) -> List[Dict]:
        """
        加载历史日志记录

        Returns:
            历史记录列表
        """
        if os.path.exists(self.json_log_file):
            with open(self.json_log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def print_summary(self):
        """打印训练摘要"""
        if not self.training_history:
            print("No training history available.")
            return

        print("\n=== 训练历史摘要 ===")
        for entry in self.training_history:
            round_num = entry['round']
            global_loss = entry['global_metrics'].get('loss', 'N/A')
            global_acc = entry['global_metrics'].get('accuracy', 'N/A')
            elapsed_time = entry.get('elapsed_time', 'N/A')
            noniid_effectiveness = entry.get('noniid_effectiveness', 'N/A')
            print(f"轮次 {round_num}: "
                  f"全局损失={global_loss}, "
                  f"全局准确率={global_acc}, "
                  f"运行时间={elapsed_time:.2f}s, "
                  f"NonIID效果={noniid_effectiveness:.4f}")
