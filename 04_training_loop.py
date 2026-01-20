"""
=======================================
PyTorch 进阶教程 2：训练循环与优化
=======================================

本教程介绍完整的训练流程，包括数据加载、损失函数、优化器、
学习率调度、训练循环等。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time


def print_section(title):
  """打印分节标题"""
  print("\n" + "=" * 50)
  print(f"  {title}")
  print("=" * 50)


# ============================================================================
# 1. 创建自定义数据集
# ============================================================================
print_section("1. 创建自定义数据集")


class CustomDataset(Dataset):
  """自定义数据集类"""

  def __init__(self, n_samples=1000, input_size=20, n_classes=5):
    # 生成随机数据
    self.X = torch.randn(n_samples, input_size)
    # 生成随机标签
    self.y = torch.randint(0, n_classes, (n_samples,))

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]


# 创建数据集
dataset = CustomDataset(n_samples=1000, input_size=20, n_classes=5)
print(f"数据集大小: {len(dataset)}")
print(f"第一个样本: x.shape={dataset[0][0].shape}, y={dataset[0][1]}")

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"批次数: {len(dataloader)}")

# 查看一个批次
X_batch, y_batch = next(iter(dataloader))
print(f"批次形状: X={X_batch.shape}, y={y_batch.shape}")

# ============================================================================
# 2. 损失函数
# ============================================================================
print_section("2. 损失函数")

# 回归损失
mse_loss = nn.MSELoss()
predictions = torch.tensor([2.5, 0.3, 1.7])
targets = torch.tensor([3.0, -0.5, 2.0])
mse = mse_loss(predictions, targets)
print(f"MSE Loss: {mse.item():.4f}")

mae_loss = nn.L1Loss()
mae = mae_loss(predictions, targets)
print(f"MAE Loss: {mae.item():.4f}")

# 分类损失
ce_loss = nn.CrossEntropyLoss()
# 分类: batch_size=3, num_classes=5
logits = torch.randn(3, 5)
labels = torch.tensor([1, 2, 0])
ce = ce_loss(logits, labels)
print(f"CrossEntropy Loss: {ce.item():.4f}")

# 二分类损失
bce_loss = nn.BCELoss()
probs = torch.sigmoid(torch.randn(3, 1))
targets_binary = torch.tensor([1.0, 0.0, 1.0]).unsqueeze(1)
bce = bce_loss(probs, targets_binary)
print(f"BCE Loss: {bce.item():.4f}")

# Hinge Loss (SVM) - 修正版
hinge_loss = nn.HingeEmbeddingLoss()
input_data = torch.randn(3)  # 修正为一维，长度为3
targets = torch.tensor([1, 1, -1], dtype=torch.float)
hinge = hinge_loss(input_data, targets)
print(f"Hinge Loss: {hinge.item():.4f}")

# ============================================================================
# 3. 优化器
# ============================================================================
print_section("3. 优化器")

# 创建一个简单模型
model = nn.Sequential(
  nn.Linear(20, 64),
  nn.ReLU(),
  nn.Linear(64, 32),
  nn.ReLU(),
  nn.Linear(32, 5)
)

print("优化器对比:")

# SGD
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(f"SGD: lr=0.01, momentum=0.9")

# Adam
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
print(f"Adam: lr=0.001")

# RMSprop
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.001)
print(f"RMSprop: lr=0.001")

# AdamW (带权重衰减的 Adam)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
print(f"AdamW: lr=0.001, weight_decay=0.01")

# ============================================================================
# 4. 学习率调度器
# ============================================================================
print_section("4. 学习率调度器")

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# StepLR: 每 step_size 个 epoch 降低 lr
scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
print("StepLR: 每 3 个 epoch 学习率 × 0.1")

# ExponentialLR: 每个 epoch 指数衰减
scheduler_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
print("ExponentialLR: 每个 epoch 学习率 × 0.95")

# CosineAnnealingLR: 余弦退火
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
print("CosineAnnealingLR: 余弦退火, T_max=10")

# ReduceLROnPlateau: 根据指标调整
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
  optimizer, mode='min', factor=0.1, patience=5
)
print("ReduceLROnPlateau: 指标 5 epoch 不改善则降低学习率")

# OneCycleLR: 单周期学习率
scheduler_onecycle = optim.lr_scheduler.OneCycleLR(
  optimizer, max_lr=0.1, total_steps=100
)
print("OneCycleLR: 单周期策略")

# ============================================================================
# 5. 完整的训练循环
# ============================================================================
print_section("5. 完整的训练循环")


# 定义模型
class SimpleClassifier(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
    self.fc3 = nn.Linear(hidden_size // 2, num_classes)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    x = self.dropout(self.relu(self.fc1(x)))
    x = self.dropout(self.relu(self.fc2(x)))
    x = self.fc3(x)
    return x


# 创建模型和数据
model = SimpleClassifier(input_size=20, hidden_size=64, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 训练和验证数据集
train_dataset = CustomDataset(n_samples=800, input_size=20, n_classes=5)
val_dataset = CustomDataset(n_samples=200, input_size=20, n_classes=5)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for X_batch, y_batch in dataloader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    # 前向传播
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 统计
    running_loss += loss.item() * X_batch.size(0)
    _, predicted = outputs.max(1)
    total += y_batch.size(0)
    correct += predicted.eq(y_batch).sum().item()

  epoch_loss = running_loss / total
  epoch_acc = correct / total
  return epoch_loss, epoch_acc


# 验证函数
def validate(model, dataloader, criterion, device):
  model.eval()
  running_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for X_batch, y_batch in dataloader:
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)

      outputs = model(X_batch)
      loss = criterion(outputs, y_batch)

      running_loss += loss.item() * X_batch.size(0)
      _, predicted = outputs.max(1)
      total += y_batch.size(0)
      correct += predicted.eq(y_batch).sum().item()

  epoch_loss = running_loss / total
  epoch_acc = correct / total
  return epoch_loss, epoch_acc


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"使用设备: {device}")

# 训练循环
num_epochs = 5
best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print(f"\n开始训练 ({num_epochs} epochs)...")
start_time = time.time()

for epoch in range(num_epochs):
  train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
  val_loss, val_acc = validate(model, val_loader, criterion, device)

  scheduler.step()

  history['train_loss'].append(train_loss)
  history['train_acc'].append(train_acc)
  history['val_loss'].append(val_loss)
  history['val_acc'].append(val_acc)

  # 保存最佳模型
  if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), 'models/best_model.pth')

  print(f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}")

training_time = time.time() - start_time
print(f"\n训练完成！用时: {training_time:.2f}秒")
print(f"最佳验证准确率: {best_val_acc:.4f}")

# ============================================================================
# 6. 梯度裁剪
# ============================================================================
print_section("6. 梯度裁剪")


# 在训练循环中使用梯度裁剪（防止梯度爆炸）
def train_with_clip(model, dataloader, criterion, optimizer, device, clip_value=1.0):
  model.train()
  for X_batch, y_batch in dataloader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)

    optimizer.zero_grad()
    loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    optimizer.step()


# ============================================================================
# 7. 早停 (Early Stopping)
# ============================================================================
print_section("7. 早停 (Early Stopping)")


class EarlyStopping:
  """早停机制"""

  def __init__(self, patience=7, min_delta=0, verbose=True):
    self.patience = patience
    self.min_delta = min_delta
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.best_model = None

  def __call__(self, val_loss, model):
    score = -val_loss

    if self.best_score is None:
      self.best_score = score
      self.save_checkpoint(model)
    elif score < self.best_score + self.min_delta:
      self.counter += 1
      if self.verbose:
        print(f'EarlyStopping 计数: {self.counter}/{self.patience}')
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_score = score
      self.save_checkpoint(model)
      self.counter = 0

  def save_checkpoint(self, model):
    """保存模型"""
    if self.verbose:
      print(f'验证损失改善，保存模型...')
    self.best_model = model.state_dict().copy()


# 使用早停的示例
print("早停使用示例:")
print("early_stopping = EarlyStopping(patience=5, verbose=True)")
print("for epoch in range(num_epochs):")
print("    train_loss, val_loss = train(...)")
print("    early_stopping(val_loss, model)")
print("    if early_stopping.early_stop:")
print("        print('触发早停')")
print("        break")

# ============================================================================
# 8. 混合精度训练 (Mixed Precision)
# ============================================================================
print_section("8. 混合精度训练")

from torch.cuda.amp import autocast, GradScaler


# 混合精度训练可以加速训练并减少显存使用
def train_mixed_precision(model, dataloader, criterion, optimizer, device):
  """使用混合精度训练"""
  model.train()
  scaler = GradScaler()  # 用于缩放损失

  for X_batch, y_batch in dataloader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    # 使用 autocast 自动处理精度
    with autocast():
      outputs = model(X_batch)
      loss = criterion(outputs, y_batch)

    optimizer.zero_grad()

    # 缩放损失并进行反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


print("混合精度训练可以:")
print("  - 加速训练（约 2-3 倍）")
print("  - 减少显存使用")
print("  - 需要支持 Tensor Core 的 GPU（Volta 架构及以上）")

# ============================================================================
# 9. 模型评估
# ============================================================================
print_section("9. 模型评估")

# 加载最佳模型
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()


# 预测
def predict(model, X, device):
  model.eval()
  with torch.no_grad():
    X = X.to(device)
    outputs = model(X)
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
  return predictions, probabilities


# 测试预测
test_X = torch.randn(5, 20)
predictions, probs = predict(model, test_X, device)
print(f"预测结果: {predictions}")
print(f"预测概率: {probs[0]}")

print("\n" + "=" * 50)
print("  进阶教程 2 完成！")
print("  练习建议：")
print("  1. 尝试不同的优化器和学习率调度策略")
print("  2. 实现完整的早停机制")
print("  3. 学习使用 TensorBoard 可视化训练过程")
print("=" * 50)
