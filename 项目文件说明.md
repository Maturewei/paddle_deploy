# PaddlePickUp：让智能零售"看得清，数得明"

## 📋 项目概述

本项目是基于飞桨PaddleDetection的智能零售商品拿取检测系统，通过AI技术实现智能售货柜的商品拿取行为识别与计数。项目通过行为级检测方案，实现了从识别商品到识别动作的技术转变，显著提升了检测准确率。

---

## 🎯 项目核心特点

- **行为级检测**：从识别商品到识别动作，提高准确率
- **多人检测**：支持单人和多人同时拿取场景
- **智能柜定位**：识别12种颜色，映射到6行×2列位置
- **实时处理**：支持视频实时检测和统计分析
- **完整部署**：提供Streamlit可视化界面，易于使用

---

## 📁 项目文件结构

```
代码文件包/
├── 📂 data_dir/                          # 数据集目录
│   ├── annotations/                      # 标注文件
│   │   ├── instance_train.json          # 训练集标注
│   │   └── instance_val.json            # 验证集标注
│   └── photos/                          # 图片数据（7533张）
│       └── *.jpg                        # 原始图像文件
│
├── 📂 gui_paddle/                       # Streamlit可视化应用（核心功能）
│   ├── app.py                          # 主应用程序（检测与统计）
│   ├── model_loader.py                 # PaddleDetection模型加载器
│   ├── multi_person_detector.py        # 多人检测模块（双FSM状态机）
│   ├── font_utils.py                   # 中文字体工具
│   ├── requirements.txt                # Python依赖包列表
│   ├── __init__.py                     # Python包标识
│   └── output_prediction.mp4           # 输出视频示例
│
├── 📂 output_inference/                 # 推理模型文件
│   ├── infer_cfg.yml                    # 推理配置文件
│   ├── model.pdmodel                    # 模型结构文件
│   └── model.pdiparams                  # 模型参数文件
│
├── 📂 PaddleDetection/                  # PaddleDetection框架
│   ├── configs/                         # 配置文件目录
│   ├── deploy/                          # 部署工具
│   ├── docs/                           # 文档
│   ├── ppdet/                          # 核心代码
│   ├── tools/                          # 训练和推理工具
│   └── requirements.txt                # 依赖列表
│
├── 📂 test/                             # 测试视频
│   ├── 单人拿取/
│   │   └── 单人.mp4
│   └── 多人拿取/
│       ├── 03同取.mp4
│       ├── 11一拿.mp4
│       ├── 8同取.mp4
│       └── output/
│           └── *.mp4
│
├── 📄 code.ipynb                        # 项目演示笔记本
├── 📄 main.ipynb                        # 主项目说明笔记本
├── 📄 可视化设计说明.md                  # UI设计文档
└── 📄 项目文件说明.md                   # 本文档
```

---

## 🔧 核心模块说明

### 1. gui_paddle/ - Streamlit可视化应用

#### app.py - 主应用程序
- **功能**：Streamlit Web应用主界面
- **特性**：
  - 模型加载和验证
  - 视频上传和处理
  - 实时检测过程显示
  - 结果可视化（热力图、统计表格）
  - 处理后视频下载
- **版本**：v3.6
- **行数**：约690行

#### model_loader.py - 模型加载器
- **功能**：PaddleDetection模型适配层
- **特性**：
  - 包装PaddleDetection Detector
  - 模拟YOLO风格API接口
  - 安全的模型加载管理
  - 支持错误处理和验证
- **行数**：约368行

#### multi_person_detector.py - 多人检测算法
- **功能**：基于双FSM状态机的多人拿取检测
- **核心特性**：
  - 双FSM独立追踪（支持最多2人）
  - IoU匹配算法
  - 动作窗口机制（80帧）
  - 冷却机制避免重复计数
  - 手臂检测阈值和手部商品检测阈值
- **行数**：约266行

#### font_utils.py - 字体工具
- **功能**：中文字体显示支持
- **特性**：
  - 自动查找系统中文字体
  - 跨平台兼容（Windows/Mac/Linux）
  - PIL/OpenCV文字绘制
  - 字体下载（备用方案）
- **行数**：约170行

### 2. data_dir/ - 数据集

#### 数据概况
- **图片数量**：7533张
- **数据来源**：河南智售宝科技有限公司合作采集
- **采集周期**：6个月
- **标注格式**：COCO格式

#### 标注类别

**人员相关类标签**（6类）：
- `hand` - 手部
- `arm` - 手臂
- `handGoods` - 手持商品

**挡板颜色类标签**（12类）：
- `argent`, `yellow`, `green`, `grey`, `orange`, `cyan`
- `black`, `scarlet`, `red`, `blue`, `purple`, `darkBlue`

**颜色位置映射**（6行×2列）：
| 行 | 第1列 | 第2列 |
|---|---|---|
| 1 | argent | black |
| 2 | yellow | scarlet |
| 3 | green | red |
| 4 | grey | blue |
| 5 | orange | purple |
| 6 | cyan | darkBlue |

### 3. output_inference/ - 推理模型

#### 模型文件组成
- **infer_cfg.yml**：推理配置（预处理、后处理、标签列表）
- **model.pdmodel**：模型结构定义
- **model.pdiparams**：模型权重参数

#### 模型导出方式
```bash
cd PaddleDetection
python tools/export_model.py \
    -c configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml \
    -o weights=your_model.pdparams
```

### 4. PaddleDetection/ - 深度学习框架

#### 目录结构
- **configs/**：各种检测模型配置文件（PP-YOLOE、Faster-RCNN等）
- **deploy/**：部署工具（Python推理、C++推理、Serving等）
- **ppdet/**：核心检测算法实现
- **tools/**：训练和推理工具
- **docs/**：文档和教程

#### 环境要求
- PaddlePaddle 2.6.2
- Python 3.10
- CUDA（可选，GPU加速）

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 进入应用目录
cd gui_paddle

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置模型路径

编辑 `gui_paddle/app.py`，修改第98行的模型路径：

```python
# 方式1：使用相对路径（推荐）
MODEL_PATH = "../output_inference"

# 方式2：使用绝对路径
# MODEL_PATH = r"D:\output_inference"
```

### 3. 启动应用

```bash
cd gui_paddle
streamlit run app.py
```

访问：http://localhost:8501

### 4. 使用流程

1. **上传视频**：选择MP4/AVI/MOV/MKV格式
2. **调整参数**（可选）：
   - 置信度阈值（默认0.5）
   - 动作窗口大小（默认80帧）
   - 手臂检测阈值（默认3次）
   - 手部商品检测阈值（默认2次）
3. **开始检测**：点击"开始视频预测"
4. **查看结果**：
   - 处理摘要（总帧数、拿取次数、处理时长）
   - 智能柜行列统计（热力图、详情记录）
   - 预测视频（在线播放）
   - 下载处理后的视频

---

## 💻 技术实现

### 检测算法

#### 双FSM状态机设计

**状态流转**：
```
idle → waiting → cooldown → idle
  ↓        ↓         ↓
手臂检测  手部商品检测  冷却期
```

**FSM1和FSM2独立工作**：
- FSM1：追踪第1个人（arm_boxes[0]）
- FSM2：追踪第2个人（arm_boxes[1]）

**关键参数**：
- `action_window=80`：动作窗口大小
- `arm_thresh=3`：手臂检测阈值
- `handGoods_thresh=2`：手部商品检测阈值
- `arm_miss_limit=4`：手臂消失限制
- `cooldown_limit=5`：冷却限制
- `iou_threshold=0.25`：IoU匹配阈值

#### 位置识别逻辑

1. **颜色检测**：检测12种挡板颜色
2. **颜色映射**：映射到6行×2列位置
3. **置信度过滤**：只保存置信度≥阈值的位置
4. **后端修正**：确保位置数≤拿取总数

### 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| PaddlePaddle | 2.6.2 | 深度学习框架 |
| PaddleDetection | 2.8.0 | 目标检测开发套件 |
| Streamlit | 1.28.0+ | Web应用框架 |
| OpenCV | 4.8.0+ | 图像处理 |
| NumPy | 1.24.0+ | 数值计算 |
| Pandas | 2.0.0+ | 数据分析 |
| Matplotlib | 3.7.0+ | 可视化 |
| PyYAML | 6.0+ | 配置文件解析 |
| Pillow | 10.0.0+ | 图像处理 |

---

## 📊 功能特性

### 核心功能

1. **单人/多人检测**
   - 双FSM状态机独立追踪
   - 自动兼容单人和多人场景
   - 智能匹配算法避免误识别

2. **智能柜行列统计**
   - 12种颜色识别
   - 6行×2列位置映射
   - 热力图可视化
   - 拿取详情记录

3. **实时可视化**
   - 逐帧检测过程显示
   - 检测框标注
   - 计数实时更新
   - 中文界面支持

4. **结果分析**
   - 拿取次数统计
   - 行列位置分析
   - 颜色分布统计
   - 置信度分析

5. **视频处理**
   - 支持多种视频格式
   - 自定义输出质量
   - 处理后视频下载
   - 检测日志导出

### 创新点

1. **硬件创新**：双柜门+挡板设计
2. **算法创新**：从识别商品到识别动作
3. **场景覆盖**：单人、多人、白天、夜间
4. **用户体验**：实时检测、直观可视化

---

## 📖 依赖包说明

### gui_paddle/requirements.txt

```txt
paddlepaddle==2.6.2           # PaddlePaddle深度学习框架
streamlit>=1.28.0            # Web应用框架
opencv-python>=4.8.0         # 图像处理
opencv-contrib-python>=4.8.0 # OpenCV扩展功能
Pillow>=10.0.0               # 图像处理
numpy>=1.24.0                # 数值计算
pandas>=2.0.0                # 数据分析
matplotlib>=3.7.0            # 可视化
PyYAML>=6.0                  # YAML解析
requests>=2.31.0             # HTTP请求
```

### 安装命令

```bash
cd gui_paddle
pip install -r requirements.txt
```

---

## 🎓 项目应用场景

### 应用领域

1. **智能零售**：无人售货柜商品拿取检测
2. **仓储物流**：货架拣货动作识别
3. **安防监控**：行为分析和异常检测
4. **教育培训**：AI技术演示工具

### 实际应用案例

- **河南智售宝科技有限公司**：实际部署数据采集
- **B站演示视频**：https://www.bilibili.com/video/BV1HEs6zfEe5/

---

## 📈 数据集说明

### 数据集统计

- **总图片数**：7533张
- **训练集**：约6026张（80%）
- **验证集**：约753张（10%）
- **测试集**：约754张（10%）

### 数据增强策略

- 多种光照条件
- 多种拍摄角度
- 单人/多人场景
- 白天/夜间场景

---

## 🔍 文件说明

### 核心文件

| 文件名 | 行数 | 功能说明 |
|--------|------|----------|
| `gui_paddle/app.py` | ~690行 | 主应用程序，集成检测和统计功能 |
| `gui_paddle/model_loader.py` | ~368行 | PaddleDetection模型适配和加载 |
| `gui_paddle/multi_person_detector.py` | ~266行 | 多人检测算法实现 |
| `gui_paddle/font_utils.py` | ~170行 | 中文字体显示工具 |

### 数据文件

| 路径 | 说明 |
|------|------|
| `data_dir/photos/` | 7533张训练图片 |
| `data_dir/annotations/instance_train.json` | 训练集标注文件 |
| `data_dir/annotations/instance_val.json` | 验证集标注文件 |

### 模型文件

| 路径 | 说明 |
|------|------|
| `output_inference/infer_cfg.yml` | 推理配置 |
| `output_inference/model.pdmodel` | 模型结构 |
| `output_inference/model.pdiparams` | 模型权重 |

### 测试文件

| 路径 | 说明 |
|------|------|
| `test/单人拿取/单人.mp4` | 单人测试视频 |
| `test/多人拿取/03同取.mp4` | 多人同取测试视频 |
| `test/多人拿取/11一拿.mp4` | 多人一拿测试视频 |

---

## 🛠️ 开发环境

### 推荐配置

- **CPU**：8核心或以上
- **内存**：16GB或以上
- **存储**：20GB或以上
- **GPU**：支持CUDA的显卡（可选）

### 开发工具

- Python 3.10
- Jupyter Notebook（用于训练和测试）
- Visual Studio Code（推荐）
- Git（版本控制）

---

## 📝 更新日志

### v3.6 (2025-10-26)
- 简化界面，删除相互印证分析展示
- 保留后端智能修正逻辑
- 提升用户体验

### v3.5 (2025-10-26)
- 添加后端智能修正算法
- 确保位置识别数≤拿取总数
- 提升数据一致性

### v3.4 (2025-10-26)
- 添加相互印证逻辑
- 置信度过滤机制
- 智能数据分析

### v3.3 (2025-10-26)
- 优化界面信息展示顺序
- 简化侧边栏显示

### v3.2 (2025-10-26)
- 简化模型配置方式
- 代码配置替代界面配置
- 提升部署效率

---

## 🎯 使用建议

### 对于开发者

1. **学习项目**：
   - 阅读 `可视化设计说明.md` 了解UI设计
   - 查看 `gui_paddle/app.py` 学习主逻辑
   - 研究 `gui_paddle/multi_person_detector.py` 理解检测算法

2. **修改配置**：
   - 编辑 `gui_paddle/app.py` 第98行修改模型路径
   - 调整参数在侧边栏进行

3. **扩展功能**：
   - 参考现有模块添加新功能
   - 保持代码风格一致

### 对于用户

1. **快速开始**：
   - 按照"快速开始"章节配置环境
   - 准备好测试视频
   - 启动应用开始使用

2. **参数调整**：
   - 根据实际场景调整置信度阈值
   - 针对不同光照条件调整参数

3. **结果分析**：
   - 关注拿取总数（主要依据）
   - 参考行列统计（位置信息）
   - 查看热力图了解拿取分布

---

## 📞 技术支持

### 相关文档

- **可视化设计说明.md**：UI设计和使用说明
- **code.ipynb**：项目演示笔记本
- **main.ipynb**：主项目说明
- **PaddleDetection文档**：框架使用指南

### 开源资源

- [PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddlePaddle官方文档](https://www.paddlepaddle.org.cn/)
- [Streamlit官方文档](https://docs.streamlit.io/)

---

## 📜 许可证

本项目遵循PaddleDetection的开源许可证。

---

## 🙏 致谢

感谢以下机构和个人的支持：

- **飞桨PaddlePaddle**：提供强大的深度学习框架
- **河南智售宝科技有限公司**：提供实际应用场景和数据支持
- **飞桨AI Studio星河社区**：提供开发和展示平台

---

**项目状态**：✅ 完成并可用于生产环境  
**推荐指数**：⭐⭐⭐⭐⭐  
**最后更新**：2025-10-27
