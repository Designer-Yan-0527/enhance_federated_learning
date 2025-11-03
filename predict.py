import torch
from PIL import Image
from torchvision import transforms
from model import ResNet18
from classes import CIFAR100_CLASSES
from config import DEVICE


def predict_image(img_path, mdl_path):
    """
    对单张图片进行预测

    参数:
        img_path: 图片路径
        mdl_path: 训练好的模型路径
    """
    # 加载模型
    model = ResNet18().to(DEVICE)
    model.load_state_dict(torch.load(mdl_path, map_location=DEVICE))
    model.eval()

    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # 加载并预处理图片
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = output.argmax(dim=1).item()

    # 显示结果
    print(f"预测类别索引: {predicted_class}")
    print(f"预测类别名称: {CIFAR100_CLASSES[predicted_class]}")
    print("各类别概率:")
    # 显示前5个最可能的类别
    top5_prob, top5_ind = torch.topk(probabilities, 5)
    for i in range(5):
        print(f"  {CIFAR100_CLASSES[top5_ind[i]]}: {top5_prob[i].item():.4f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("使用方法: python predict_cifar100.py <图片路径> <模型路径>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2]
    predict_image(image_path, model_path)
