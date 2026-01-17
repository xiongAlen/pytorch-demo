# PyTorch 从入门到实战教程

这是一个完整的 PyTorch 学习项目，包含从基础到进阶的系统教程，以及 MNIST 手写数字识别实战项目。

## 项目结构

```
pytorch_demo/
├── data/                   # 数据集存储目录
├── models/                 # 模型保存目录
├── checkpoints/            # 训练检查点目录
├── 01_tensor_basics.py     # 基础教程1：张量操作
├── 02_autograd.py          # 基础教程2：自动求导
├── 03_neural_network.py    # 进阶教程1：构建神经网络
├── 04_training_loop.py     # 进阶教程2：训练循环与优化
├── 05_mnist_project.py     # 实战项目：MNIST分类
├── requirements.txt        # 项目依赖
└── README.md              # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n pytorch python=3.9
conda activate pytorch

# 安装依赖
pip install -r requirements.txt
```

### 2. 学习路径

按照以下顺序学习，循序渐进：

#### 第一阶段：基础教程

1. **运行基础教程1：张量操作**
   ```bash
   python 01_tensor_basics.py
   ```
   学习内容：张量的创建、索引、切片、数学运算、GPU使用

2. **运行基础教程2：自动求导**
   ```bash
   python 02_autograd.py
   ```
   学习内容：梯度计算、计算图、链式法则、优化器使用

#### 第二阶段：进阶教程

3. **运行进阶教程1：构建神经网络**
   ```bash
   python 03_neural_network.py
   ```
   学习内容：nn.Module、常用层、CNN、模型初始化

4. **运行进阶教程2：训练循环与优化**
   ```bash
   python 04_training_loop.py
   ```
   学习内容：数据加载、损失函数、优化器、学习率调度、早停

#### 第三阶段：实战项目

5. **运行实战项目：MNIST手写数字识别**
   ```bash
   python 05_mnist_project.py
   ```
   综合应用：完整的深度学习项目流程

## 教程内容概览

### 01_tensor_basics.py - 张量基础

- 张量的创建与初始化
- 形状操作（reshape、transpose、flatten）
- 索引与切片
- 数学运算（矩阵乘法、广播）
- 与 NumPy 互转
- GPU/CUDA 支持

### 02_autograd.py - 自动求导

- 梯度计算基础
- 理解计算图
- 多元函数的梯度
- 禁用梯度计算
- 梯度积累与清零
- 实现梯度下降
- 使用优化器

### 03_neural_network.py - 构建神经网络

- nn.Module 的使用
- 常用层（Linear、Conv2d、Pool、BatchNorm、Dropout）
- 激活函数
- 构建多层感知机 (MLP)
- 构建卷积神经网络 (CNN)
- 残差连接
- 参数初始化
- 模型保存与加载

### 04_training_loop.py - 训练循环

- 自定义数据集 (Dataset)
- 数据加载器 (DataLoader)
- 损失函数 (MSE、CrossEntropy、BCE)
- 优化器 (SGD、Adam、AdamW)
- 学习率调度器
- 完整的训练循环
- 早停机制
- 混合精度训练

### 05_mnist_project.py - MNIST 实战项目

- 完整的项目配置
- 数据增强
- CNN 模型设计
- 训练与验证
- 模型保存与加载
- 预测与评估
- 详细的结果分析

## 学习建议

### 对于初学者

1. **不要跳过基础**：确保理解张量和自动求导，这是 PyTorch 的核心
2. **动手实践**：运行每个文件，观察输出，修改参数看效果
3. **逐步深入**：完成一个教程再进入下一个

### 对于有一定基础的学习者

1. **尝试修改代码**：改变网络结构、超参数，观察效果变化
2. **解决实际问题**：将学到的知识应用到自己的项目中
3. **阅读官方文档**：深入了解 PyTorch 的高级特性

## 常见问题

### Q: 如何选择 GPU 还是 CPU？

代码会自动检测并使用可用的 GPU：
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Q: 如何在 Apple Silicon (M1/M2) Mac 上使用 GPU？

PyTorch 支持 MPS (Metal Performance Shaders)：
```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

### Q: 训练时显存不足怎么办？

1. 减小 batch_size
2. 使用梯度累积
3. 使用混合精度训练
4. 减小模型规模

### Q: 如何提高训练速度？

1. 使用 DataLoader 的 num_workers 参数
2. 使用 pin_memory=True（GPU 训练时）
3. 使用混合精度训练
4. 增加 batch_size（在显存允许范围内）

## 进阶学习资源

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [PyTorch 中文文档](https://pytorch-cn.readthedocs.io/)

## 许可证

MIT License

## 作者

PyTorch 学习项目 Demo

---

祝学习愉快！如有问题，欢迎交流讨论。
