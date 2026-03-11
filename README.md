<div align="center">

# 🚀 ABSNet: Adaptive Rank Spectral Network for Lightweight Domain Generalization in Hyperspectral Image Classification
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/zhulongyu1234/ARSNet/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

</div>

Note: The full model code (model.py) will be publicly released upon paper acceptance.
Currently, we provide the trained weights, logs, and classification maps.

## Dataset
You can download the datasets on [here](https://github.com/YuxiangZhang-BIT/Data-CSHSI). 
The dataset directory should look like this:
```
datasets
├── Houston
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
├── Pavia
│   ├── paviaC.mat
│   ├── paviaC_7gt.mat
│   ├── paviaU.mat
│   └── paviaU_7gt.mat
└── Hyrank
│   ├── Dioni.mat
│   ├── Dioni_gt_out68.mat
│   ├── Loukia.mat
│   └── Loukia_gt_out68.mat
```
## Requirement
* CUDA Version: 11.8
* PyTorch version: 2.2.0+cu118
* Python version: 3.10.18
## Usage
* You can run the `main.py` with `python main.py`

## Quantitative Results

| Dataset       | OA (%) ↑ | AA (%) ↑ | Kappa (%) ↑ | Params (K) ↓ |
|---------------|----------|----------|-------------|--------------|
| Houston | **78.15** | **64.36** | **62.88** | **3.74** |
| Pavia   | **87.19** | **85.72** | **84.57** | **5.71** |
| HyRank  | **68.60** | **48.50** | **61.30** | **8.43** |

> All results are obtained using **only source-domain labels**.  
> Classification maps, training logs, and trained checkpoints are provided in the `results/` folder.

## Acknowledgment
 Our code is based on the method of [ACB](https://github.com/zhulongyu1234/ACB/tree/master). Thanks for their work.
