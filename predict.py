import torch
from PIL import Image
from torchvision import transforms
from model import ResNet18
from classes import CIFAR10_CLASSES
from config import DEVICE


def predict_image(image_path, model_path):
    """
    对单张图片进行预测

    参数:
        image_path: 图片路径
        model_path: 训练好的模型路径
    """
    # 加载模型
    model = ResNet18().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载并预处理图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = output.argmax(dim=1).item()

    # 显示结果
    print(f"预测类别索引: {predicted_class}")
    print(f"预测类别名称: {CIFAR10_CLASSES[predicted_class]}")
    print("各类别概率:")
    for i, prob in enumerate(probabilities):
        print(f"  {CIFAR10_CLASSES[i]}: {prob.item():.4f}")
