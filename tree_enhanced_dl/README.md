# 树增强深度学习框架

一个专业的 PyTorch 框架，将梯度提升树与深度神经网络相结合，用于表格数据分类任务，内置可解释性模块并支持不平衡数据集处理。

## 核心特性

- **树增强架构**：深度整合 LightGBM/XGBoost/CatBoost 与深度学习模型
- **自动化特征工程**：从树模型中提取交叉特征和路径序列
- **多损失优化**：融合交叉熵、对比损失、AUC 损失和焦点损失
- **不平衡数据处理**：类别平衡采样与代价敏感学习策略
- **可解释性**：注意力权重、规则贡献度和特征重要性分析
- **模块化设计**：组件易于定制和扩展
- **生产级就绪**：完善的日志记录、断点续训和推理流水线

## 安装指南

```bash
# 克隆代码仓库
git clone https://github.com/yourusername/tree_enhanced_dl.git
cd tree_enhanced_dl

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows 系统执行: venv\Scripts\activate

# 安装依赖包
pip install -r requirements.txt
```

### 1. 配置准备

编辑 `configs/default_config.yaml` 文件，指定数据路径和超参数：

```yaml
train_path: "data/train.csv"
val_path: "data/val.csv"
target_column: "label"
numerical_features: ["年龄", "收入", ...]
categorical_features: ["性别", "城市", ...]
```

### 2. 模型训练

```bash
python main.py
```

使用自定义参数训练：

```bash
python main.py \
    --config configs/default_config.yaml \
    --batch-size 256 \
    --epochs 100 \
    --lr 0.001
```

### 3. 推理预测

```bash
python inference.py \
    --checkpoint-dir checkpoints \
    --input-file data/test.csv \
    --output-file predictions.csv \
    --return-proba
```

## 项目结构

```bash
├── configs/              # 配置文件目录
├── data/                 # 数据处理模块
│   ├── preprocessor.py   # 特征预处理
│   ├── dataset.py        # PyTorch 数据集定义
│   └── sampler.py        # 不平衡数据采样器
├── tree_module/          # 树模型组件
│   ├── tree_trainer.py   # 树模型训练
│   ├── rule_extractor.py # 规则提取
│   └── path_encoder.py   # 路径编码
├── models/               # 神经网络模型
│   ├── embeddings.py     # 嵌入层
│   ├── encoders.py       # 序列编码器
│   ├── fusion.py         # 融合模块
│   ├── heads.py          # 分类头
│   └── tree_enhanced_model.py  # 完整模型定义
├── losses/               # 损失函数
│   ├── contrastive_loss.py # 对比损失
│   ├── ranking_loss.py   # 排序损失
│   └── combined_loss.py  # 组合损失
├── training/             # 训练工具
│   ├── trainer.py        # 训练循环
│   └── scheduler.py      # 学习率调度器
├── evaluation/           # 评估与解释
│   ├── metrics.py        # 指标计算
│   └── explainer.py      # 模型解释器
├── utils/                # 通用工具
│   ├── config_parser.py  # 配置解析
│   └── logger.py         # 日志工具
├── main.py               # 训练主脚本
├── inference.py          # 推理脚本
└── requirements.txt      # 依赖清单
```

## 配置说明

### 数据配置

```yaml
missing_value_strategy: "median"        # 缺失值处理策略：中位数填充
numerical_normalization: "standard"    # 数值特征归一化：标准化
categorical_encoding: "ordinal"        # 类别特征编码：序数编码
sampling_strategy: "class_balanced"    # 采样策略：类别平衡
```

### 模型架构

```yaml
sequence_encoder:
  type: "bilstm"  # 可选：transformer / gru
fusion:
  type: "multi_head_attention"  # 可选：concat / gated
```

### 损失配置

```yaml
inner:
  ce_weight: 1.0           # 交叉熵损失权重
  contrastive_weight: 0.5  # 对比损失权重
outer:
  auc_weight: 0.3          # AUC损失权重
  focal_weight: 0.5        # 焦点损失权重
```

## 高级用法

### 自定义数据预处理

```python
preprocessor = DataPreprocessor(config)
X_transformed, y_encoded = preprocessor.fit_transform(df)
```

### 模型解释

```python
explainer = ModelExplainer(model, config, device)
explanation = explainer.explain_instance(batch, sample_idx=0)
```

### 自定义损失函数

```python
loss_fn = CombinedLoss(config, num_classes=2, feature_dim=128)
losses = loss_fn(logits, labels, features)
```

## 评估指标

框架内置以下评估指标：

* ROC 曲线下面积 (ROC AUC)
* 精确率-召回率曲线下面积 (PR AUC，适用于不平衡数据)
* F1 分数
* KS 统计量
* 期望校准误差 (ECE)

## 可解释性分析

* **注意力权重**：可视化原始特征、路径特征和交叉特征的重要性
* **规则贡献度**：识别每个预测结果触发的树规则
* **特征重要性**：全局和局部特征重要性评分
* **路径分析**：解析树集成模型的决策路径

## 性能优化

### 大规模数据集优化

```yaml
batch_size: 512
data:
  num_workers: 8          # 数据加载线程数
deployment:
  batch_inference: true   # 开启批量推理
  inference_batch_size: 1024  # 推理批次大小
```

### 加速训练

```yaml
n_estimators: 50  # 减少树模型数量
model:
  sequence_encoder:
    num_layers: 1  # 减少编码器层数
```

## 训练监控

### TensorBoard 监控

```bash
tensorboard --logdir runs/
```

### Weights & Biases 集成

```yaml
logging:
  wandb:
    enabled: true         # 启用W&B监控
    project: "tree_enhanced_dl"  # 项目名称
```

## 引用格式

如果您在研究中使用了本框架，请引用：

```bibtex
@misc{tree_enhanced_dl_2024,
  title={树增强深度学习框架},
  author={你的姓名},
  year={2024},
  url={https://github.com/yourusername/tree_enhanced_dl}
}
```

## 开源协议

MIT 开源协议 - 详见 LICENSE 文件

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交修改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

## 技术支持

如有问题或建议：

* GitHub Issues: [https://github.com/yourusername/tree_enhanced_dl/issues](https://github.com/yourusername/tree_enhanced_dl/issues)
* 邮箱: [your.email@example.com](mailto:your.email@example.com)

### .gitignore 文件

```gitignore
# Python 相关
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
venv/
env/
ENV/

# IDE 配置
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter 笔记本
.ipynb_checkpoints/
*.ipynb

# 数据文件
data/
*.csv
*.parquet
*.pkl
*.h5

# 模型和日志
checkpoints/
logs/
runs/
wandb/
*.pt
*.pth
*.txt.meta

# 系统文件
.DS_Store
Thumbs.db

# 临时文件
tmp/
temp/
*.tmp
```

---

## 完整项目总结

本树增强深度学习框架具备以下核心价值：

### 核心特性

1. **模块化设计** - 各组件可独立替换和扩展，降低维护成本
2. **配置驱动** - 所有参数通过 YAML 文件统一管理，便于实验复现
3. **生产级就绪** - 包含完整的训练、推理、监控和解释能力

### 核心组件

* **数据处理** (`data/`) - 预处理、数据集构建、不平衡采样
* **树模块** (`tree_module/`) - 树模型训练、规则提取、路径编码
* **模型层** (`models/`) - 嵌入层、序列编码器、特征融合、分类头
* **损失函数** (`losses/`) - 对比损失、排序损失、多目标组合损失
* **训练引擎** (`training/`) - 训练循环、学习率调度、早停机制
* **评估体系** (`evaluation/`) - 多维度指标计算、模型可解释性分析

### 标准使用流程

1. 编写配置文件定义数据路径和超参数
2. 运行 `main.py` 启动模型训练
3. 通过 `inference.py` 进行批量推理
4. 查看日志和可视化分析结果

# Tree-Enhanced Deep Learning Framework

A professional PyTorch framework for combining gradient boosting trees with deep neural networks for tabular data classification, with built-in interpretability and handling of imbalanced datasets.

## Features

- **Tree-Enhanced Architecture**: Integrates LightGBM/XGBoost/CatBoost with deep learning
- **Automatic Feature Engineering**: Extracts cross features and path sequences from trees
- **Multi-Loss Optimization**: Combines CE, contrastive, AUC, and focal losses
- **Imbalance Handling**: Class-balanced sampling and cost-sensitive learning
- **Interpretability**: Attention weights, rule contributions, and feature importance
- **Modular Design**: Easy to customize and extend components
- **Production Ready**: Comprehensive logging, checkpointing, and inference pipeline

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/tree_enhanced_dl.git
cd tree_enhanced_dl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 1. Prepare Configuration

Edit `configs/default_config.yaml` to specify your data paths and hyperparameters:

```bash<pre><div
  train_path: "data/train.csv"
  val_path: "data/val.csv"
  target_column: "label"
  numerical_features: ["age", "income", ...]
  categorical_features: ["gender", "city", ...]
</code></div></div></pre>
```

### 2. Train Model

```bash<pre><div
</code></div></div></pre>
```

With custom parameters:

```bash<pre><div
    --config configs/default_config.yaml \
    --batch-size 256 \
    --epochs 100 \
    --lr 0.001
</code></div></div></pre>
```

### 3. Run Inference

```bash<pre><div
    --checkpoint-dir checkpoints \
    --input-file data/test.csv \
    --output-file predictions.csv \
    --return-proba
</code></div></div></pre>
```

## Project Structure

```bash<pre><div
├── configs/              # Configuration files
├── data/                 # Data processing modules
│   ├── preprocessor.py   # Feature preprocessing
│   ├── dataset.py        # PyTorch datasets
│   └── sampler.py        # Imbalanced data samplers
├── tree_module/          # Tree model components
│   ├── tree_trainer.py   # Tree training
│   ├── rule_extractor.py # Rule extraction
│   └── path_encoder.py   # Path encoding
├── models/               # Neural network models
│   ├── embeddings.py     # Embedding layers
│   ├── encoders.py       # Sequence encoders
│   ├── fusion.py         # Fusion modules
│   ├── heads.py          # Classification heads
│   └── tree_enhanced_model.py  # Complete model
├── losses/               # Loss functions
│   ├── contrastive_loss.py
│   ├── ranking_loss.py
│   └── combined_loss.py
├── training/             # Training utilities
│   ├── trainer.py        # Training loop
│   └── scheduler.py      # LR schedulers
├── evaluation/           # Evaluation and interpretation
│   ├── metrics.py        # Metrics calculation
│   └── explainer.py      # Model explanation
├── utils/                # Utilities
│   ├── config_parser.py
│   └── logger.py
├── main.py               # Training script
├── inference.py          # Inference script
└── requirements.txt
</code></div></div></pre>
```

## Configuration

Key configuration sections:

### Data Configuration

```bash<pre><div
  missing_value_strategy: "median"
  numerical_normalization: "standard"
  categorical_encoding: "ordinal"
  sampling_strategy: "class_balanced"
</code></div></div></pre>
```

### Model Architecture

```bash<pre><div
  sequence_encoder:
    type: "bilstm"  # or "transformer", "gru"
  fusion:
    type: "multi_head_attention"  # or "concat", "gated"
</code></div></div></pre>
```

### Loss Configuration

```bash<pre><div
  inner:
    ce_weight: 1.0
    contrastive_weight: 0.5
  outer:
    auc_weight: 0.3
    focal_weight: 0.5
</code></div></div></pre>
```

## Advanced Usage

### Custom Data Preprocessing

```bash<pre><div

preprocessor = DataPreprocessor(config)
X_transformed, y_encoded = preprocessor.fit_transform(df)
</code></div></div></pre>
```

### Model Explanation

```bash<pre><div

explainer = ModelExplainer(model, config, device)
explanation = explainer.explain_instance(batch, sample_idx=0)
</code></div></div></pre>
```

### Custom Loss Function

```bash<pre><div

loss_fn = CombinedLoss(config, num_classes=2, feature_dim=128)
losses = loss_fn(logits, labels, features)
</code></div></div></pre>
```

## Evaluation Metrics

The framework computes:

* ROC AUC
* PR AUC (for imbalanced data)
* F1 Score
* KS Statistic
* Expected Calibration Error (ECE)

## Interpretability

* **Attention Weights** : Visualize importance of original, path, and cross features
* **Rule Contributions** : Identify which tree rules fire for each prediction
* **Feature Importance** : Global and local feature importance scores
* **Path Analysis** : Understand decision paths through tree ensemble

## Performance Optimization

### For Large Datasets

```bash<pre><div
  batch_size: 512
data:
  num_workers: 8
deployment:
  batch_inference: true
  inference_batch_size: 1024
</code></div></div></pre>
```

### For Faster Training

```bash<pre><div
  n_estimators: 50  # Reduce trees
model:
  sequence_encoder:
    num_layers: 1  # Reduce layers
</code></div></div></pre>
```

## Monitoring

### TensorBoard

```bash<pre><div
</code></div></div></pre>
```

### Weights & Biases

```bash<pre><div
  logging:
    wandb:
      enabled: true
      project: "tree_enhanced_dl"
</code></div></div></pre>
```

## Citation

If you use this framework in your research, please cite:

```bash<pre><div
  title={Tree-Enhanced Deep Learning Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/tree_enhanced_dl}
}
</code></div></div></pre>
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:

* GitHub Issues: [https://github.com/yourusername/tree_enhanced_dl/issues](https://github.com/yourusername/tree_enhanced_dl/issues)
* Email: [your.email@example.com](mailto:your.email@example.com)

```bash<pre><div
### 26. `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Data
data/
*.csv
*.parquet
*.pkl
*.h5

# Models and logs
checkpoints/
logs/
runs/
wandb/
*.pt
*.pth
*.txt.meta

# OS
.DS_Store
Thumbs.db

# Temporary files
tmp/
temp/
*.tmp
</code></div></div></pre>
```

---

## 完整项目总结

这个专业的树增强深度学习框架包含：

### 核心特性

1. **模块化设计** - 每个组件都可以独立替换和扩展
2. **配置驱动** - 所有参数通过 YAML 配置文件统一管理
3. **生产就绪** - 包含完整的训练、推理、监控和解释功能

### 主要组件

* **数据处理** (`data/`) - 预处理、数据集、采样器
* **树模块** (`tree_module/`) - 树模型训练、规则提取、路径编码
* **模型** (`models/`) - 嵌入层、编码器、融合模块、分类头
* **损失函数** (`losses/`) - 对比学习、排序、组合损失
* **训练** (`training/`) - 训练循环、学习率调度
* **评估** (`evaluation/`) - 指标计算、模型解释

### 使用流程

1. 准备配置文件
2. 运行 `main.py` 训练模型
3. 使用 `inference.py` 进行推理
4. 查看日志和可视化结果
