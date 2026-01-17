"""
=======================================
PyTorch 基础教程 1：张量 (Tensor) 操作
=======================================

张量是 PyTorch 的核心数据结构，类似于 NumPy 的数组，但可以在 GPU 上运行。
本教程涵盖张量的创建、操作、索引等基础知识。
"""

import torch
import numpy as np


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)


# ============================================================================
# 1. 张量的创建
# ============================================================================
print_section("1. 张量的创建")

# 从列表创建张量
t1 = torch.tensor([1, 2, 3])
print(f"从列表创建: {t1}")

# 创建指定形状的张量（全零）
t2 = torch.zeros(2, 3)
print(f"全零张量 (2x3):\n{t2}")

# 创建指定形状的张量（全一）
t3 = torch.ones(2, 3)
print(f"全一张量 (2x3):\n{t3}")

# 创建随机张量（0-1 之间的均匀分布）
t4 = torch.rand(2, 3)
print(f"随机张量 (2x3):\n{t4}")

# 创建随机整数张量
t5 = torch.randint(0, 10, (2, 3))
print(f"随机整数张量 (2x3):\n{t5}")

# 从 NumPy 数组创建
np_array = np.array([[1, 2], [3, 4]])
t6 = torch.from_numpy(np_array)
print(f"从 NumPy 创建:\n{t6}")

# 指定数据类型和设备
t7 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device='cpu')
print(f"指定数据类型: {t7.dtype}, 设备: {t7.device}")


# ============================================================================
# 2. 张量的形状操作
# ============================================================================
print_section("2. 张量的形状操作")

# 创建一个 2x3 的张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"原始张量 (2x3):\n{x}")

# 获取形状
print(f"形状: {x.shape}")  # 或 x.size()

# 改变形状 (reshape)
x_reshaped = x.reshape(3, 2)
print(f"Reshape 为 (3x2):\n{x_reshaped}")

# 展平
x_flattened = x.flatten()
print(f"展平: {x_flattened}")

# 增加维度
x_unsqueezed = x.unsqueeze(0)  # 在第 0 维增加
print(f"增加维度后形状: {x_unsqueezed.shape}")

# 转置
x_transposed = x.T  # 或 x.t()
print(f"转置 (3x2):\n{x_transposed}")


# ============================================================================
# 3. 张量的索引和切片
# ============================================================================
print_section("3. 张量的索引和切片")

x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
print(f"原始张量 (3x4):\n{x}")

# 获取单个元素
print(f"x[0, 0] = {x[0, 0].item()}")  # .item() 获取 Python 标量

# 获取一行
print(f"第 0 行: {x[0]}")

# 获取一列
print(f"第 0 列: {x[:, 0]}")

# 切片
print(f"前两行，前两列:\n{x[:2, :2]}")

# 布尔索引
mask = x > 5
print(f"大于 5 的元素: {x[mask]}")

# 高级索引
rows = torch.tensor([0, 2])
cols = torch.tensor([1, 3])
print(f"选取 (0,1) 和 (2,3): {x[rows, cols]}")


# ============================================================================
# 4. 张量的数学运算
# ============================================================================
print_section("4. 张量的数学运算")

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
print(f"张量 A:\n{a}")
print(f"张量 B:\n{b}")

# 基本运算
print(f"A + B:\n{a + b}")
print(f"A - B:\n{a - b}")
print(f"A * B (逐元素乘法):\n{a * b}")
print(f"A / B:\n{a / b}")

# 矩阵乘法
print(f"A @ B (矩阵乘法):\n{a @ b}")
print(f"torch.matmul(A, B):\n{torch.matmul(a, b)}")
print(f"torch.mm(A, B):\n{torch.mm(a, b)}")

# 广播机制
c = torch.tensor([[1], [2], [3]])  # (3, 1)
d = torch.tensor([4, 5, 6])        # (3,)
print(f"C (3x1) + D (3) = (3x3):\n{c + d}")

# 常用数学函数
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
print(f"原始: {x}")
print(f"平方: {x ** 2}")
print(f"开方: {torch.sqrt(x)}")
print(f"指数: {torch.exp(x)}")
print(f"对数: {torch.log(x)}")

# 统计运算
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(f"求和: {x.sum()}")
print(f"均值: {x.mean()}")
print(f"标准差: {x.std()}")
print(f"按行求和: {x.sum(dim=1)}")
print(f"按列求和: {x.sum(dim=0)}")
print(f"每行最大值及索引: {x.max(dim=1)}")


# ============================================================================
# 5. 张量与 NumPy 互转
# ============================================================================
print_section("5.1 张量与 NumPy 互转")

# 张量转 NumPy
t = torch.tensor([1, 2, 3])
np_arr = t.numpy()
print(f"张量转 NumPy: {np_arr}, 类型: {type(np_arr)}")

# NumPy 转张量
np_arr = np.array([4, 5, 6])
t = torch.from_numpy(np_arr)
print(f"NumPy 转张量: {t}, 类型: {type(t)}")

# 注意：它们共享内存
t[0] = 999
print(f"修改张量后，NumPy 数组也会变: {np_arr}")

print_section("5.2 张量与 NumPy 互转（深拷贝版本）")

# 张量转 NumPy（深拷贝）
t = torch.tensor([1, 2, 3])
np_arr = t.clone().numpy()  # 使用 clone() 避免内存共享
print(f"张量转 NumPy (深拷贝): {np_arr}, 类型: {type(np_arr)}")

# 修改张量，验证 NumPy 数组不受影响
t[0] = 999
print(f"修改张量后，NumPy 数组不变: {np_arr}")  # 不受影响

# NumPy 转张量（深拷贝）
np_arr = np.array([4, 5, 6])
t = torch.tensor(np_arr.copy())  # 使用 copy() 避免内存共享
print(f"NumPy 转张量 (深拷贝): {t}, 类型: {type(t)}")

# 修改 NumPy 数组，验证张量不受影响
np_arr[0] = 888
print(f"修改 NumPy 后，张量不变: {t}")  # 不受影响

# ============================================================================
# 6. GPU 支持
# ============================================================================
print_section("6. GPU 支持")

# 检查 CUDA 是否可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 将张量移动到 GPU
    x_cpu = torch.tensor([1, 2, 3])
    x_gpu = x_cpu.to('cuda')  # 或 x_cpu.cuda()
    print(f"CPU 张量设备: {x_cpu.device}")
    print(f"GPU 张量设备: {x_gpu.device}")

    # 在 GPU 上创建张量
    y_gpu = torch.tensor([4, 5, 6], device='cuda')
    print(f"直接在 GPU 创建: {y_gpu.device}")

    # GPU 上的运算
    z_gpu = x_gpu + y_gpu
    print(f"GPU 运算结果设备: {z_gpu.device}")

    # 移回 CPU
    z_cpu = z_gpu.to('cpu')  # 或 z_gpu.cpu()
    print(f"移回 CPU: {z_cpu.device}")
else:
    print("当前环境不支持 CUDA，使用 MPS (Apple Silicon) 或 CPU")
    # 检查 MPS (Apple Silicon GPU)
    print(f"MPS 是否可用: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        x_mps = torch.tensor([1, 2, 3]).to('mps')
        print(f"MPS 张量设备: {x_mps.device}")


# ============================================================================
# 7. 实用技巧
# ============================================================================
print_section("7. 实用技巧")

# 设置随机种子（保证可复现）
torch.manual_seed(42)
print(f"设置随机种子后的随机数: {torch.rand(3)}")

# 生成相同的随机数
torch.manual_seed(42)
print(f"相同种子，相同随机数: {torch.rand(3)}")

# 梯度相关（后续章节详细讲解）
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"需要梯度的张量: {x.requires_grad}")

# 类型转换
x = torch.tensor([1, 2, 3])
print(f"原始类型: {x.dtype}")
print(f"转为 float: {x.float().dtype}")
print(f"转为 long: {x.long().dtype}")

# 保存和加载张量
x = torch.tensor([1, 2, 3, 4, 5])
torch.save(x, 'tensor.pt')
loaded = torch.load('tensor.pt')
print(f"保存并加载张量: {loaded}")

# 清理临时文件
import os
os.remove('tensor.pt')


print("\n" + "=" * 50)
print("  基础教程 1 完成！")
print("  练习建议：")
print("  1. 尝试创建不同形状的张量并进行运算")
print("  2. 熟悉张量的索引和切片操作")
print("  3. 理解广播机制的工作原理")
print("=" * 50)
