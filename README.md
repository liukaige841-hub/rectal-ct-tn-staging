# 直肠 CT 自动分割与 T/N 分期：精简研究代码

本包仅包含最终模型定义、训练与推理、预处理、关键层选择、ROI 构建、固定配置、测试和论文可用性声明。T 与 N 是两个独立二分类模型，不是共享的多任务模型。

**本包不附带任何原始影像、掩膜、患者级文件或训练权重。已完成语法及随机数组流程检查；完整模型前向测试需在安装 PyTorch 后执行，真实病例端到端测试还需另行提供匹配权重。代码仅供研究，不可直接用于临床诊断或治疗决策。**

## 1. 最终流程与模型

1. **SEG-2D-UNET**：2D U-Net，输入 `1×256×256`，基础通道数 32。CT 按腹窗 WL 50 / WW 350，即 HU `[-125, 225]` 窗化；每层 `rot90`、双线性缩放、除以 255。输出概率**严格 `>0.5`** 才为前景，随后以最近邻还原尺寸并逆旋转。
2. **固定后处理**：单层面积小于 20 视为噪声；允许 2 层 z 向空隙；保留体素数最大的 z 连续段；向上下各外延 2 层并清除段外区域；段内仅保留 26 邻接最大三维连通域。
3. **2.5D ROI**：后处理掩膜横截面积最大层为关键层；关键层上下各 8 层的掩膜并集确定 xy 边界框，宽、高分别向两侧按 0.4 倍外扩；裁剪并双线性缩放至 `224×224`。`z-1/z/z+1` 三张灰度 ROI 组成三通道，z 边界采用复制。最终模型仅使用 2.5D，不包含 3D 分支。
4. **T-BIN-AUTO / N-BIN-AUTO**：ResNet152 加 `dual_branch`。在 `layer1` 后，一个分支使用大核最大池化，另一个分支使用中心裁剪；两个分支各产生 128 维特征，与主干 2048 维特征拼成 2304 维，再经过 `Linear(2304,512) → ReLU → Dropout(0.3) → Linear(512,2)`。输入使用 ImageNet 三通道均值和方差归一化。
5. **输出**：T 正类为 `P(T≥3)`，N 正类为 `P(N+)`；二者均在概率 **`>=0.5`** 时预测正类。注意分割是严格 `>0.5`，分类是 `>=0.5`。

### 空掩膜差异（重要）

- **推理**：分割或后处理后为空掩膜时，关键层取 `Z//2`，ROI 取显示平面的中心 50% 裁剪。
- **训练侧 ROI 构建**：空掩膜时使用清单中的人工 GT 关键层 `key_z`，并使用中心 50% 裁剪；缺少 `key_z` 会报错。这一训练/推理差异必须在使用和解释结果时保留。

## 2. 文件结构

```text
README.md
MODEL_CARD.md
CODE_AND_DATA_AVAILABILITY.md
requirements.txt
run_inference.py
configs/
  segmentation.yaml
  T_staging.yaml
  N_staging.yaml
  inference.yaml
src/
  segmentation.py          # 2D U-Net、窗化、逐层推理、损失
  models.py                # ResNet152 + dual_branch
  imaging.py               # DICOM/NIfTI 读取
  postprocess.py           # 固定掩膜后处理
  roi.py                   # 关键层、边界框、2.5D ROI、归一化
  pipeline.py              # 完整推理流程
  build_segmentation_dataset.py
  build_roi_dataset.py
  datasets.py
  train_segmentation.py
  train_classifier.py
  inference_utils.py
  config.py
tests/
  smoke_test.py
```

## 3. 环境与测试

建议使用独立 Python 环境。在本目录执行：

```bash
python -m pip install -r requirements.txt
python -m compileall -q src run_inference.py tests
python tests/smoke_test.py
```

冒烟测试只需要随机数组和随机张量，不需要影像、掩膜或训练权重。分类训练配置使用 ImageNet 预训练初始化；若本机没有缓存，首次构建可能需要获取相应预训练参数。纯结构测试使用 `pretrained=false`，不会要求下载。

## 4. 一键推理

输入可以是：

- 一个三维 `.nii` 或 `.nii.gz` 文件，体数据应为 HU，读取后转换到 canonical 方向；
- 一个包含单例 CT DICOM 的目录，程序递归读取、选择实例数最多的 CT 序列、应用 rescale slope/intercept，并转换为 canonical 体数据。

```bash
python run_inference.py \
  --input /path/to/series_or_volume \
  --seg-weights /path/to/segmentation.pt \
  --t-weights /path/to/T-BIN-AUTO.pt \
  --n-weights /path/to/N-BIN-AUTO.pt \
  --config configs/inference.yaml \
  --output /path/to/prediction.json
```

三个参数都必填，且必须分别传入与公开结构匹配的权重。公开包不含权重；文件缺失、checkpoint 格式不符或 `state_dict` 不匹配时会明确失败，不会生成虚构概率或类别。

可选保存三张灰度 ROI：

```bash
python run_inference.py \
  --input /path/to/series_or_volume \
  --seg-weights /path/to/segmentation.pt \
  --t-weights /path/to/T-BIN-AUTO.pt \
  --n-weights /path/to/N-BIN-AUTO.pt \
  --output /path/to/prediction.json \
  --save-roi-dir /path/to/roi_preview
```

JSON 包含 T/N 正类概率、冻结阈值、预测类别，以及关键层、空掩膜兜底标记和掩膜体素计数。`predicted_class=1` 表示正类，`0` 表示负类。默认不保存原始影像或掩膜；显式指定 `--save-roi-dir` 才会保存 ROI。

## 5. 训练数据准备

所有路径均通过命令行或 YAML 提供。清单中的 `sample_id` 应使用研究内脱敏标识；不要在公开清单、输出或日志中写入直接身份标识。

### 5.1 分割

准备 CSV，每行一例：

```text
sample_id,image_path,mask_path,split
```

`image_path` 为 HU NIfTI，`mask_path` 为同几何肿瘤标注 NIfTI，`split` 为 `train` 或 `val`。相对路径相对于清单所在目录。训练集与验证集应按病例隔离。

```bash
python -m src.build_segmentation_dataset \
  --manifest /path/to/segmentation_manifest.csv \
  --output-dir /path/to/segmentation_slices \
  --config configs/segmentation.yaml

python -m src.train_segmentation \
  --config configs/segmentation.yaml \
  --index-csv /path/to/segmentation_slices/index.csv \
  --output-dir /path/to/segmentation_training
```

训练固定参数为：`img_size=256`、`base=32`、60 epochs、batch 32、Adam `lr=3e-4`、patience 10、seed 42、`neg_frac=1.0`、`hard_neg_weight=2.0`、加权 BCE + soft Dice，以 `val_dice_pos` 保存最佳权重。困难负样本清单是可选的运行输入，不包含在本包中；需要时用 `--hard-negative-csv` 指定含 `image_path` 列的 CSV。

### 5.2 T 与 N 分类

分别准备 T 和 N 的 ROI 构建清单：

```text
sample_id,image_path,mask_path,label,split,key_z
```

其中 `mask_path` 指与训练流程一致的后处理分割掩膜；`label` 已映射为最终二分类标签：T 的 1 表示 T≥3，N 的 1 表示 N+；`key_z` 是人工 GT 关键层，仅用于空掩膜训练兜底。原始标签定义和任何临床明细不属于本公开包。

```bash
python -m src.build_roi_dataset \
  --manifest /path/to/T_manifest.csv \
  --output-dir /path/to/T_roi \
  --config configs/T_staging.yaml

python -m src.train_classifier \
  --config configs/T_staging.yaml \
  --labels-csv /path/to/T_roi/labels.csv \
  --output-dir /path/to/T_training

python -m src.build_roi_dataset \
  --manifest /path/to/N_manifest.csv \
  --output-dir /path/to/N_roi \
  --config configs/N_staging.yaml

python -m src.train_classifier \
  --config configs/N_staging.yaml \
  --labels-csv /path/to/N_roi/labels.csv \
  --output-dir /path/to/N_training
```

T、N 分别训练，不共享 checkpoint。共同训练配置为 ResNet152、`dual_branch=true`、二分类、`224×224`、邻层数 1、ImageNet 预训练初始化、batch 32、Adam `lr=2e-5`、`weight_decay=1e-4`、CosineAnnealingLR、100 epochs、early-stop patience 15、AMP、balanced sampler、无 class weight、label smoothing 0.1、seed 42。T 的 `min_epochs=10`，N 的 `min_epochs=15`。

## 6. 权重格式

- 分割 checkpoint：推荐包含 `model`（UNet `state_dict`）、`base=32`、`img_size=256`。
- 分类 checkpoint：推荐包含 `state_dict`（对应 `DualBranchResNet`）和训练配置。

加载采用严格参数匹配。权重属于不可信输入时，请先核验来源和完整性；不要加载来源不明的文件。

## 7. 使用限制

- 输入协议、重建核、层厚、增强期相和人群分布变化可能影响结果。
- 多序列 DICOM 目录仅自动选择实例数最多的 CT 序列；正式研究前应人工确认目标序列。
- 空掩膜兜底不是肿瘤定位证据，输出中的 `empty_mask_fallback=true` 必须纳入质控。
- 本包没有公开性能指标，也不暗示外部泛化能力。真实研究使用需在授权数据上复现预处理、训练、验证和质控。
