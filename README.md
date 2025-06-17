# SoftLabel Steganalysis Network (StegoNet)



[](@replace=1)

> 基于深度学习的软标签隐写分析网络，实现图像隐写检测与嵌入率预测的多任务学习

##  项目概述
本项目开发了一个多任务深度学习模型（StegoNet），用于检测图像中的LSB隐写信息并预测嵌入率。创新性地采用：
- **软标签技术**：将连续嵌入率转换为概率分布标签
- **多任务学习**：并行处理分类（区间预测）和回归（精确预测）任务
- **高通滤波增强**：专门设计的卷积核增强微弱信号特征

**核心创新**：
- 融合高通滤波与SE注意力机制
- 多任务损失函数设计
- 温度缩放软标签技术
  
##  项目结构
```
src/
├── config.py # 训练参数配置
├── train.py # 主训练脚本
├── model/
│ ├── net.py # StegoNet模型架构
│ ├── multitask_loss.py # 多任务损失函数
│ └── soft_label_utils.py # 软标签转换工具
├── dataset/
│ ├── clean.py # 数据清洗
│ ├── dataset.py # 数据集类
│ ├── process&loader.py # 隐写处理与加载
│ ├── readable.py # 可视化工具
│ └── split.py # 数据集划分
result/ # 训练结果与可视化
```
##  项目开启
### 环境要求
- Python 3.8+
- PyTorch 1.12+
- OpenCV 4.5+

### 安装依赖
```bash
```bash
conda create -n stegonet python=3.8
conda install pytorch==1.12 torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```
## 训练流程
### 数据准备
```bash
# 处理BossBase和BOWS2数据集
python dataset/split.py       # 数据集划分
python dataset/process&loader.py  # 生成隐写图像
```
### 模型训练
```bash
python train.py --config config.py \
  --data_dir ./processed_data \
  --output_dir ./results
```
## 核心组件
### 网络架构
- 高通滤波层​​：8个固定核（5标准+3LSB专用）
- SE注意力模块​​：通道级特征重标定
- 双输出头​​：分类头（11类区间预测） 回归头（[0,1]连续值预测）

### 核心组件
```python
total_loss = 0.7*KLDivLoss(softmax(logits/T), soft_labels) + 0.3*MSELoss(reg_pred, true_rate)
#温度参数T=2用于标签平滑
```
### 数据处理流程
| 模块               | 功能         | 关键技术           |
|--------------------|--------------|--------------------|
| `split.py`         | 数据集划分    | 随机打乱，标准命名 |
| `process&loader.py`| LSB隐写处理  | 比特操作，批量处理 |
| `StegoDataset`     | 数据加载      | 自动标签提取       |

## 实验结果
### 性能指标
| 指标 | 训练集 | 验证集 |
|------|--------|--------|
| 准确率 | 87.43% | 85.12% |
| MAE | 0.035 | 0.038 |
| MSE | 0.0028 | 0.0031 |

### 嵌入率预测
| 真实α | 预测均值±标准差 |
|-------|-----------------|
| 0.0 | 0.163±0.110 |
| 0.5 | 0.503±0.055 |
| 1.0 | 0.839±0.105 |
