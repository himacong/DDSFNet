# Saliency-Guided Spatial–Frequency Dependency Modelling for Infrared and Visible Image Fusion
## Introduction
Infrared and visible image fusion (IVIF) is a fundamental task in multimodal visual computing, which aims to integrate complementary information from two heterogeneous imaging modalities into a single high-quality fused image. 
Visible images capture rich colour details, fine textures and high spatial resolution, delivering superior visual perception under normal lighting conditions.
Nevertheless, their imaging quality degrades drastically in challenging scenarios such as low illumination, fog and smoke. By contrast, infrared images record thermal radiation emitted by objects, enabling reliable detection of heat-emitting targets like pedestrians and vehicles regardless of adverse weather and lighting. However, infrared imagery inherently suffers from blurry textures and low spatial resolution.

## Main contributions
(1) Cross-modal global-local dependency learning: A Dual-Dependent Efficient Interaction Module (DDEIM) is designed to combine attention blocks and patch-wise multi-layer perceptrons (MLP) for joint capture of global channel correlations and local structural interactions between infrared and visible features, strengthening dynamic cross-modal feature communication.
(2) Spatial-frequency collaborative enhancement: A Spatial-Frequency Combine Module (SFCM) is designed for multi-scale spatial feature extraction. And Fourier transformation are integrated to realise bidirectional optimisation of spatial textures and frequency-domain edge information. This design effectively suppresses noise interference caused by independent frequency-domain operations.
(3) Saliency-aware target preservation: Saliency masks generated from infrared images are leveraged to build a lightweight guidance mechanism for adaptive feature weight allocation, which ensures that thermal targets remain salient while fully retaining the rich texture details of visible images.
(4) Comprehensive task validation: Extensive evaluations are conducted on four mainstream public benchmarks. Additional validation on object detection tasks fully verifies the generalisation ability and practical value of the proposed framework for downstream visual perception applications.


## Dataset description
- **MSRS**：A road-scene dataset adopted for model training, covering complex environments with day/night cycles and varying strong/weak illumination conditions.
- **M3FD**：A multi-modal multispectral dataset equipped with complete object detection annotations, utilized for fusion evaluation and downstream detection task validation.
- **RoadScene**：A vehicle-mounted scene dataset dedicated to driving-oriented image fusion tests.
- **TNO**：A classic general-purpose dataset containing diverse scenarios such as cities, villages and harbors, which is used to verify the generalization capability of the proposed model.

## SOTA algorithm
DIDFuse、U2Fusion、YDTR、TarDal、SFDFusion。

## Installation


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
├── configs/               
│   ├── cfg.yaml           
│   └── yolov8s.yaml       
├── dataset.py             
├── modules.py             
├── utils/               
│   ├── evaluator.py        
│   ├── get_params_group.py 
│   ├── img_read.py       
│   ├── loss.py           
│   ├── misc.py           
│   ├── plot_labels.py    
│   ├── saliency.py      
│   ├── u2net.py         
│   └── u2netp.pth       
├── train.py            
├── val.py             
├── fuse.py            
└── README.md          

