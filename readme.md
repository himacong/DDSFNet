# Saliency-Guided Spatial–Frequency Dependency Modelling for Infrared and Visible Image Fusion
## Introduction
本仓库为论文 **DDSFNet: 一种用于红外光与可见光图像融合的双依赖空频联合感知融合网络** 对应的官方 PyTorch 代码实现。

针对现有红外与可见光图像融合方法存在**跨模态特征交互不足、高频细节易丢失**，复杂场景下目标纹理模糊、显著性特征缺失等问题，本文提出**双依赖空间频率联合感知融合网络（DDSFNet）**。网络采用双分支并行架构，结合空域依赖建模与频域信息增强，并引入显著特征引导机制，有效提升多模态特征互补能力。在多个公开数据集上的实验表明，该方法在视觉效果与量化指标上均优于主流算法，同时在下游目标检测任务中具备良好的应用价值。

## Main contributions
### Cross-modal global-local dependency learning: A Dual-Dependent Efficient Interaction Module (DDEIM) is designed to combine attention blocks and patch-wise multi-layer perceptrons (MLP) for joint capture of global channel correlations and local structural interactions between infrared and visible features, strengthening dynamic cross-modal feature communication.
### Spatial-frequency collaborative enhancement: A Spatial-Frequency Combine Module (SFCM) is designed for multi-scale spatial feature extraction. And Fourier transformation are integrated to realise bidirectional optimisation of spatial textures and frequency-domain edge information. This design effectively suppresses noise interference caused by independent frequency-domain operations.
### Saliency-aware target preservation: Saliency masks generated from infrared images are leveraged to build a lightweight guidance mechanism for adaptive feature weight allocation, which ensures that thermal targets remain salient while fully retaining the rich texture details of visible images.
### Comprehensive task validation: Extensive evaluations are conducted on four mainstream public benchmarks. Additional validation on object detection tasks fully verifies the generalisation ability and practical value of the proposed framework for downstream visual perception applications.


## Dataset description
- **MSRS**：道路场景数据集，用于模型训练，覆盖昼夜、强弱光照等复杂环境；
- **M3FD**：多模态多光谱数据集，含完整目标检测标注，用于融合测试与下游检测任务验证；
- **RoadScene**：车载场景数据集，面向行车视觉融合测试；
- **TNO**：经典通用数据集，包含城市、乡村、港口等多类场景，验证模型泛化能力。

SOTA algorithm ：DIDFuse、U2Fusion、YDTR、TarDal、SFDFusion。

## 环境依赖

python = 3.10
pytorch = 2.1.0
torchvision
opencv-python
numpy
scipy
pillow
kornia
tqdm
pyyaml
thop
```
可通过以下命令安装依赖：
```bash
pip install -r requirements.txt
```

## 仓库目录结构
DDSFNet/
├── configs/                # 配置文件目录
│   ├── cfg.yaml            # 基础配置（如数据集路径、训练超参）
│   └── yolov8s.yaml        # 目标检测模型结构配置
├── dataset.py              #数据集加载与预处理：自定义 Dataset 类、数据增强、数据加载器（Dataloader）
├── modules.py              # 模型模块            
├── utils/                  # 工具函数目录：如日志记录、指标计算、模型保存 / 加载、可视化等通用函数
│   ├── evaluator.py        # 模型评估相关
│   ├── get_params_group.py # 参数分组          
│   ├── img_read.py         # 图像读取相关工具函数       
│   ├── loss.py             # 损失函数定义
│   ├── misc.py             # 杂项工具函数
│   ├── plot_labels.py      # 标签可视化、绘图相关函数
│   ├── saliency.py         # 显著性检测相关工具/算法
│   ├── u2net.py            # U2Net模型相关实现
│   └── u2netp.pth          # U2Netp模型预训练权重文件
├── train.py                # 训练脚本
├── val.py                  # 验证/测试脚本
├── fuse.py                 # 融合逻辑脚本
└── README.md               #项目说明

