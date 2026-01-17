"""
=======================================
PyTorch 实战项目：MNIST 手写数字识别
=======================================

这是一个完整的深度学习项目，涵盖从数据准备到模型部署的全流程。
项目目标：构建一个 CNN 模型识别 MNIST 手写数字，准确率达到 99% 以上。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os


# ============================================================================
# 配置参数
# ============================================================================
class Config:
    # 数据配置
    batch_size = 128
    num_workers = 4

    # 模型配置
    num_classes = 10
    input_channels = 1

    # 训练配置
    num_epochs = 10
    learning_rate = 0.001
    weight_decay = 1e-4

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 路径配置
    data_dir = './data'
    checkpoint_dir = './checkpoints'
    model_name = 'mnist_cnn.pth'

config = Config()


# ============================================================================
# 数据准备
# ============================================================================
def get_data_loaders():
    """获取训练和验证数据加载器"""

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    # 下载数据集
    train_dataset = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.MNIST(
        root=config.data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


# ============================================================================
# 模型定义
# ============================================================================
class MNIST_CNN(nn.Module):
    """MNIST 手写数字识别 CNN 模型"""

    def __init__(self):
        super().__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 第二个卷积块
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # 池化
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Conv Block 1: 28x28 -> 14x14
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Conv Block 2: 14x14 -> 7x7
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# ============================================================================
# 训练和验证函数
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ============================================================================
# 早停类
# ============================================================================
class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=7, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping 计数: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0


# ============================================================================
# 主训练函数
# ============================================================================
def train():
    """主训练函数"""

    print("=" * 50)
    print("  MNIST 手写数字识别 - 训练开始")
    print("=" * 50)
    print(f"\n配置:")
    print(f"  设备: {config.device}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  训练轮数: {config.num_epochs}")

    # 创建目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 加载数据
    print("\n加载数据...")
    train_loader, test_loader = get_data_loaders()
    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"测试样本: {len(test_loader.dataset)}")

    # 创建模型
    print("\n创建模型...")
    model = MNIST_CNN().to(config.device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # 早停
    early_stopping = EarlyStopping(patience=5, verbose=True)

    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    # 训练循环
    print("\n开始训练...\n")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )

        # 验证
        val_loss, val_acc = validate(
            model, test_loader, criterion, config.device
        )

        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                config.model_name
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)

        # 打印进度
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{config.num_epochs}] "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n触发早停，停止训练！")
            break

    total_time = time.time() - start_time
    print(f"\n训练完成！")
    print(f"总用时: {total_time/60:.1f}分钟")
    print(f"最佳验证准确率: {best_val_acc:.4f}")

    # 加载最佳模型并评估
    print("\n加载最佳模型进行最终评估...")
    checkpoint = torch.load(
        os.path.join(config.checkpoint_dir, config.model_name)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    final_val_loss, final_val_acc = validate(
        model, test_loader, criterion, config.device
    )
    print(f"最终测试损失: {final_val_loss:.4f}")
    print(f"最终测试准确率: {final_val_acc:.4f}")

    return model, history


# ============================================================================
# 预测函数
# ============================================================================
def predict(model, image, device):
    """预测单个图像"""

    model.eval()
    with torch.no_grad():
        image = image.to(device)
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)

        output = model(image)
        probabilities = F.softmax(output, dim=1)
        prediction = output.argmax(dim=1)

    return prediction.item(), probabilities.squeeze().cpu()


# ============================================================================
# 模型评估详细分析
# ============================================================================
def evaluate_detailed(model, test_loader, device):
    """详细的模型评估"""

    model.eval()

    all_predictions = []
    all_targets = []
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    # 每个类别的准确率
    print("\n各类别准确率:")
    for i in range(10):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  数字 {i}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':

    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 训练模型
    model, history = train()

    # 详细评估
    _, test_loader = get_data_loaders()
    evaluate_detailed(model, test_loader, config.device)

    print("\n" + "=" * 50)
    print("  项目完成！")
    print("  模型已保存至: checkpoints/mnist_cnn.pth")
    print("=" * 50)
