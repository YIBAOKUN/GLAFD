# GLAFD: Global-Local Attention-based Fraud Detection Network

基于全局-局部注意力的欺诈检测网络，面向异配图场景的节点级欺诈检测方法。

## 项目简介

GLAFD 针对欺诈检测场景中图数据的**异配性**、**多关系异质性**和**类别不平衡**三大核心挑战，提出了一种双路径图神经网络框架。

## 环境依赖

```
Python       >= 3.9
PyTorch      >= 1.12
DGL          >= 0.9
torch-geometric
scikit-learn
numpy
scipy
```

安装依赖：

```bash
pip install torch dgl torch-geometric scikit-learn numpy scipy
```

## 项目结构

- `src/` : includes all code scripts.
  - `model.py` : Model definition of GLAFD.
  - `train.py` : Training and evaluation script for a single dataset.
  - `ablation.py` : Script for running ablation experiments.
  - `utils.py` : Data loading utilities.
  - `data_preprocess.py` : Data preprocessing script.
- `data/` : includes original datasets.
  - `YelpChi.zip` : The original dataset of YelpChi, which contains hotel and restaurant reviews filtered (spam) and recommended (legitimate) by Yelp.
  - `Amazon.zip` : The original dataset of Amazon, which contains product reviews under the Musical Instruments category.
  - `FDCompCN.zip` : The processed dataset of FDCompCN, which contains financial statement fraud of companies in China from CSMAR database.
- `config/` : includes the setting of parameters for three datasets.
  - `yelp.yaml` : The general parameters of YelpChi.
  - `amazon.yaml` : The general parameters of Amazon.
  - `comp.yaml` : The general parameters of FDCompCN.
- `results/` : includes the results of models.
- `README.md` : Project documentation.

## 数据集

数据集文件放置于 `data/` 目录下，若 `.dgl` 文件不存在，训练时将自动调用 `data_preprocess.py` 进行预处理。

## 快速开始

### 训练

```bash
# YelpChi
python src/train.py --dataset yelp --lr 0.01 --epochs 1000

# Amazon
python src/train.py --dataset amazon --lr 0.001 --epochs 1000

# FDCompCN
python src/train.py --dataset comp --lr 0.001 --epochs 1000
```

### 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | yelp | 数据集选择：yelp / amazon / comp |
| `--hidden_dim` | 64 | 节点嵌入维度 |
| `--num_layers` | 2 | GAT 层数 |
| `--num_heads` | 4 | 注意力头数 |
| `--walk_length` | 10 | 随机游走长度 |
| `--num_walks` | 5 | 每节点游走次数 |
| `--p` | 2.0 | 游走返回概率参数 |
| `--q` | 0.5 | 游走出入概率参数 |
| `--dropout` | 0.1 | Dropout 比例 |
| `--lr` | 0.001 | 学习率 |
| `--epochs` | 1000 | 训练轮数 |
