"""
=======================================
PyTorch 进阶教程 1：构建神经网络
=======================================

本教程介绍如何使用 nn.Module 构建神经网络，包括层、激活函数、
初始化等核心概念。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def print_section(title):
  """打印分节标题"""
  print("\n" + "=" * 50)
  print(f"  {title}")
  print("=" * 50)


# ============================================================================
# 1. nn.Module 基础
# ============================================================================
print_section("1. nn.Module 基础")


# 所有神经网络模块都继承自 nn.Module
class SimpleNet(nn.Module):
  def __init__(self):
    super().__init__()
    # 在 __init__ 中定义层

  def forward(self, x):
    # 在 forward 中定义前向传播
    return x


# 创建模型实例
model = SimpleNet()
print(f"模型: {model}")
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# 2. 常用层
# ============================================================================
print_section("2. 常用层")

# 全连接层 (Linear)
linear = nn.Linear(in_features=5, out_features=3)
x = torch.randn(2, 5)  # batch_size=2, input_dim=5
output = linear(x)
print(f"Linear 层:")
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {output.shape}")
print(f"  权重形状: {linear.weight.shape}")
print(f"  偏置形状: {linear.bias.shape}")

# 卷积层 (Conv2d)
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
x = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
output = conv(x)
print(f"\nConv2d 层:")
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {output.shape}")

# 池化层 (MaxPool2d)
pool = nn.MaxPool2d(kernel_size=2, stride=2)
output = pool(x)
print(f"\nMaxPool2d 层:")
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {output.shape}")

# 批归一化层 (BatchNorm2d)
bn = nn.BatchNorm2d(num_features=16)
x = torch.randn(4, 16, 32, 32)
output = bn(x)
print(f"\nBatchNorm2d 层:")
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {output.shape}")

# Dropout 层
dropout = nn.Dropout(p=0.5)
x = torch.randn(4, 10)
output = dropout(x)
print(f"\nDropout 层 (p=0.5):")
print(f"  输入: {x[0, :3]}")
print(f"  输出: {output[0, :3]}")

# ============================================================================
# 3. 激活函数
# ============================================================================
print_section("3. 激活函数")

x = torch.linspace(-3, 3, 10)

# ReLU
relu = nn.ReLU()
print(f"ReLU: {relu(x)}")

# Sigmoid
sigmoid = nn.Sigmoid()
print(f"Sigmoid: {sigmoid(x)}")

# Tanh
tanh = nn.Tanh()
print(f"Tanh: {tanh(x)}")

# LeakyReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.1)
print(f"LeakyReLU: {leaky_relu(x)}")

# Softmax
softmax = nn.Softmax(dim=0)
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Softmax: {softmax(x)}")
print(f"和为 1: {softmax(x).sum()}")

# ============================================================================
# 4. 构建完整的神经网络
# ============================================================================
print_section("4. 构建完整的神经网络")


class MLP(nn.Module):
  """多层感知机 (Multi-Layer Perceptron)"""

  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()
    # 定义层
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.bn1 = nn.BatchNorm1d(hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
    self.bn2 = nn.BatchNorm1d(hidden_size // 2)
    self.fc3 = nn.Linear(hidden_size // 2, num_classes)
    self.dropout = nn.Dropout(0.3)
    self.relu = nn.ReLU()

  def forward(self, x):
    # 第一层
    x = self.fc1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.dropout(x)

    # 第二层
    x = self.fc2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.dropout(x)

    # 输出层
    x = self.fc3(x)
    return x


# 创建模型
model = MLP(input_size=784, hidden_size=256, num_classes=10)
print(f"MLP 模型结构:\n{model}")

# 测试前向传播
x = torch.randn(4, 784)  # batch_size=4
output = model(x)
print(f"\n输入形状: {x.shape}")
print(f"输出形状: {output.shape}")

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")

# ============================================================================
# 5. 卷积神经网络 (CNN)
# ============================================================================
print_section("5. 卷积神经网络 (CNN)")


class SimpleCNN(nn.Module):
  """简单的卷积神经网络"""

  def __init__(self, num_classes=10):
    super().__init__()

    # 卷积块
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    # 池化层
    self.pool = nn.MaxPool2d(2, 2)

    # 全连接层
    self.fc1 = nn.Linear(128 * 4 * 4, 256)  # 假设输入是 32x32
    self.fc2 = nn.Linear(256, num_classes)

    # BatchNorm
    self.bn1 = nn.BatchNorm2d(32)
    self.bn2 = nn.BatchNorm2d(64)
    self.bn3 = nn.BatchNorm2d(128)

    # Dropout
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    # Conv Block 1
    x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16

    # Conv Block 2
    x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8

    # Conv Block 3
    x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4

    # 展平
    x = x.view(x.size(0), -1)  # (batch_size, 128*4*4)

    # 全连接层
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    return x


# 创建 CNN 模型
cnn = SimpleCNN(num_classes=10)
print(f"CNN 模型结构:\n{cnn}")

# 测试前向传播
x = torch.randn(2, 3, 32, 32)  # batch_size=2, 3 channels, 32x32 image
output = cnn(x)
print(f"\n输入形状: {x.shape}")
print(f"输出形状: {output.shape}")

# ============================================================================
# 6. 序列模型 (Sequential)
# ============================================================================
print_section("6. 序列模型 (Sequential)")

# 使用 nn.Sequential 快速构建模型
model = nn.Sequential(
  nn.Linear(784, 256),
  nn.ReLU(),
  nn.Dropout(0.3),
  nn.Linear(256, 128),
  nn.ReLU(),
  nn.Dropout(0.3),
  nn.Linear(128, 10)
)

print(f"Sequential 模型:\n{model}")

x = torch.randn(2, 784)
output = model(x)
print(f"输出形状: {output.shape}")

# ============================================================================
# 7. 自定义层的组合
# ============================================================================
print_section("7. 自定义层的组合")


class ResidualBlock(nn.Module):
  """残差块 (ResNet Block)"""

  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                           stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    # 跳跃连接的投影
    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1,
                  stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(x))
    out += self.shortcut(x)  # 残差连接
    out = F.relu(out)
    return out


# 测试残差块
block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
x = torch.randn(1, 64, 32, 32)
output = block(x)
print(f"Residual Block:")
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {output.shape}")

# ============================================================================
# 8. 参数初始化
# ============================================================================
print_section("8. 参数初始化")


def init_weights(m):
  """自定义初始化函数"""
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    nn.init.zeros_(m.bias)
  elif isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if m.bias is not None:
      nn.init.zeros_(m.bias)


# 应用初始化
model = MLP(784, 256, 10)
model.apply(init_weights)

# 查看初始化后的参数
for name, param in model.named_parameters():
  print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")

# ============================================================================
# 9. 查看模型信息
# ============================================================================
print_section("9. 查看模型信息")

# 模型摘要
from torchsummary import summary

try:
  # 需要 pip install torchsummary
  print("\n使用 torchsummary 查看模型摘要:")
  # summary(cnn, (3, 32, 32))
except ImportError:
  print("提示: 安装 torchsummary 可以查看更详细的模型信息")
  print("      pip install torchsummary")


# 打印每一层的输出形状
def print_layer_output(model, input_size):
  """打印每层的输出形状"""
  x = torch.randn(1, *input_size)
  print(f"\n输入形状: {x.shape}")

  for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d,
                           nn.BatchNorm2d, nn.ReLU)):
      # 这里简化处理，实际使用时需要更复杂的逻辑
      pass


# 手动打印关键层
print("\nCNN 模型的层信息:")
print(f"输入: (batch, 3, 32, 32)")
print(f"Conv1+Pool: (batch, 32, 16, 16)")
print(f"Conv2+Pool: (batch, 64, 8, 8)")
print(f"Conv3+Pool: (batch, 128, 4, 4)")
print(f"Flatten: (batch, 2048)")
print(f"FC1: (batch, 256)")
print(f"FC2: (batch, 10)")

# ============================================================================
# 10. 保存和加载模型
# ============================================================================
print_section("10. 保存和加载模型")

# 保存整个模型
torch.save(model, 'models/full_model.pth')

# 只保存模型参数（推荐）
torch.save(model.state_dict(), 'models/model_weights.pth')

# 加载模型
loaded_model = MLP(784, 256, 10)
loaded_model.load_state_dict(torch.load('models/model_weights.pth'))
loaded_model.eval()  # 设置为评估模式

print("模型已保存和加载")
print(f"加载的模型状态: {'training' if loaded_model.training else 'evaluation'} 模式")

print("\n" + "=" * 50)
print("  进阶教程 1 完成！")
print("  练习建议：")
print("  1. 尝试设计不同的网络架构")
print("  2. 理解 BatchNorm 和 Dropout 的作用")
print("  3. 学习残差连接和跳跃连接的设计思想")
print("=" * 50)
