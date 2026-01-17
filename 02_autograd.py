"""
=======================================
PyTorch 基础教程 2：自动求导 (Autograd)
=======================================

PyTorch 的自动求导系统是训练神经网络的核心。
本教程涵盖梯度计算、计算图、梯度积累等关键概念。
"""

import torch


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)


# ============================================================================
# 1. 梯度基础
# ============================================================================
print_section("1. 梯度基础")

# 创建需要计算梯度的张量
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"张量 x: {x}, requires_grad: {x.requires_grad}")

# 进行运算
y = x ** 2 + 3 * x
print(f"y = x² + 3x = {y}")

# 计算梯度（反向传播）
# dy/dx = 2x + 3
# 当 x = [2, 3] 时，dy/dx = [2*2+3, 2*3+3] = [7, 9]
y.sum().backward()  # 对 y 求和后再求导
print(f"dy/dx = {x.grad}")


# ============================================================================
# 2. 理解计算图
# ============================================================================
print_section("2. 理解计算图")

# 构建计算图
x = torch.tensor(2.0, requires_grad=True)
a = x + 1      # a = 2 + 1 = 3
b = a ** 2     # b = 3² = 9
c = b * 3      # c = 9 * 3 = 27
y = c - 1      # y = 27 - 1 = 26

print(f"计算过程: x={x.item()} -> a={a.item()} -> b={b.item()} -> c={c.item()} -> y={y.item()}")

# 查看梯度函数（构建的计算图）
print(f"y 的梯度函数: {y.grad_fn}")
print(f"a 的梯度函数: {a.grad_fn}")

# 反向传播
y.backward()

# 链式法则: dy/dx = dy/dc * dc/db * db/da * da/dx
#        = 1 * 3 * 2a * 1 = 6a = 6 * 3 = 18
print(f"dy/dx = {x.grad}")
print(f"验证链式法则: 6 * a = {6 * a.item()}")


# ============================================================================
# 3. 多元函数的梯度
# ============================================================================
print_section("3. 多元函数的梯度")

# 多输入函数
x = torch.tensor([1.0, 2.0], requires_grad=True)
W = torch.tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)
b = torch.tensor([1.0, 1.0], requires_grad=False)

# 线性变换: y = xW + b
y = x @ W.T + b
print(f"x = {x}")
print(f"W = {W}")
print(f"y = xW + b = {y}")

# 对 y 求和后反向传播
loss = y.sum()
loss.backward()

print(f"\n∂loss/∂x = {x.grad}")
print(f"∂loss/∂W = {W.grad}")


# ============================================================================
# 4. 禁用梯度计算
# ============================================================================
print_section("4. 禁用梯度计算")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 方法 1: 使用 torch.no_grad()
with torch.no_grad():
    y = x ** 2
    print(f"在 no_grad 中, y.requires_grad = {y.requires_grad}")

# 方法 2: 使用 detach()
z = x.detach()
print(f"使用 detach, z.requires_grad = {z.requires_grad}")

# 方法 3: 直接设置 requires_grad=False
x.requires_grad_(False)
print(f"设置 False 后, x.requires_grad = {x.requires_grad}")


# ============================================================================
# 5. 梯度积累
# ============================================================================
print_section("5. 梯度积累")

x = torch.tensor(2.0, requires_grad=True)

# 第一次计算
y1 = x ** 2
y1.backward()
print(f"第一次: x² 在 x=2 处的梯度 = {x.grad.item()}")

# 第二次计算（梯度会累积！）
y2 = x ** 3
y2.backward()
print(f"第二次: 累积梯度 = {x.grad.item()}")
print(f"预期: 2x + 3x² = {2*2 + 3*2**2} = {2*2 + 3*4}")

# 清零梯度
x.grad.zero_()
print(f"清零后: {x.grad}")


# ============================================================================
# 6. 实际应用：线性回归的梯度下降
# ============================================================================
print_section("6. 实际应用：线性回归的梯度下降")

# 生成数据
torch.manual_seed(42)
X = torch.randn(100, 1) * 10
true_w, true_b = 2.5, 5.0
y = true_w * X + true_b + torch.randn(100, 1) * 2

print(f"真实参数: w = {true_w}, b = {true_b}")

# 初始化参数
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 超参数
learning_rate = 0.001
epochs = 100

print(f"\n初始参数: w = {w.item():.4f}, b = {b.item():.4f}")
print("开始训练...\n")

# 梯度下降训练
for epoch in range(epochs):
    # 前向传播
    y_pred = w * X + b

    # 计算损失 (MSE)
    loss = ((y_pred - y) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新参数（手动实现）
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()

    # 打印进度
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, "
              f"w: {w.item():.4f}, b: {b.item():.4f}")

print(f"\n训练后参数: w = {w.item():.4f}, b = {b.item():.4f}")


# ============================================================================
# 7. 使用优化器
# ============================================================================
print_section("7. 使用优化器 (SGD)")

# 重新初始化参数
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 创建优化器
optimizer = torch.optim.SGD([w, b], lr=0.001)

print(f"初始参数: w = {w.item():.4f}, b = {b.item():.4f}")
print("使用 SGD 优化器训练...\n")

for epoch in range(epochs):
    # 前向传播
    y_pred = w * X + b

    # 计算损失
    loss = ((y_pred - y) ** 2).mean()

    # 反向传播
    loss.backward()

    # 优化器更新参数（会自动清零梯度）
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, "
              f"w: {w.item():.4f}, b: {b.item():.4f}")

print(f"\n训练后参数: w = {w.item():.4f}, b = {b.item():.4f}")


# ============================================================================
# 8. 高级特性：梯度裁剪
# ============================================================================
print_section("8. 梯度裁剪 (防止梯度爆炸)")

x = torch.tensor(2.0, requires_grad=True)
y = x ** 10  # 指数增长会导致很大梯度
y.backward()

print(f"未裁剪的梯度: {x.grad.item():.4f}")

# 梯度裁剪
x.grad.zero_()
y = x ** 10
y.backward()
torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
print(f"裁剪后的梯度: {x.grad.item():.4f}")


# ============================================================================
# 9. 检查梯度
# ============================================================================
print_section("9. 检查梯度（数值验证）")

def f(x):
    return x ** 3 + 2 * x ** 2 + x

x = torch.tensor(2.0, requires_grad=True)
y = f(x)
y.backward()

print(f"自动求导: f'(2) = {x.grad.item():.4f}")

# 数值梯度（有限差分法）
h = 1e-5
numerical_grad = (f(x + h) - f(x - h)) / (2 * h)
print(f"数值梯度: f'(2) ≈ {numerical_grad.item():.4f}")

# 使用 PyTorch 的梯度检查
from torch.autograd import gradcheck

# 创建一个需要检查梯度的函数
def func(x):
    return x ** 3

# gradcheck 需要输入是 double 类型的张量
input = torch.randn(2, 1, dtype=torch.double, requires_grad=True)
test = gradcheck(func, input, eps=1e-6, atol=1e-4)
print(f"梯度检查结果: {'通过' if test else '失败'}")


print("\n" + "=" * 50)
print("  基础教程 2 完成！")
print("  练习建议：")
print("  1. 理解计算图和链式法则")
print("  2. 掌握梯度积累和清零的时机")
print("  3. 尝试实现不同的损失函数")
print("=" * 50)
