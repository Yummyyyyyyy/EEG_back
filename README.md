# EEG Data Processing API

EEG数据处理、增强与分类预测后端系统

## 项目结构

```
EEG_back/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI应用入口
│   ├── config.py            # 配置文件
│   ├── api/                 # API路由
│   ├── models/              # Pydantic数据模型
│   ├── services/            # 业务逻辑层
│   ├── ml_models/           # 机器学习模型
│   ├── data/                # 数据存储
│   │   ├── raw/             # 原始数据
│   │   ├── processed/       # 处理后数据
│   │   └── cache/           # 缓存
│   └── utils/               # 工具函数
├── tests/                   # 测试文件
├── requirements.txt         # 依赖包
├── .env.example            # 环境变量示例
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量（可选）

```bash
cp .env.example .env
# 根据需要编辑 .env 文件
```

### 3. 启动开发服务器

```bash
# 方法1: 使用 uvicorn 命令
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 方法2: 直接运行 main.py
python -m app.main
```

### 4. 访问API文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/api/health

## 配置说明

### CORS配置

默认允许以下前端源访问：
- http://localhost:3001 (前端开发服务器)
- http://127.0.0.1:3001
- http://localhost:3000
- http://127.0.0.1:3000

如需修改，请编辑 `app/config.py` 中的 `CORS_ORIGINS` 配置。

### EEG数据配置

- **采样率**: 250 Hz
- **通道**: C3, Cz, C4, CP1, CP2 (5个通道)
- **受试者**: 1-9 (9个受试者)
- **运动类型**: left, right, foot, tongue (4种)
- **增强方法**: VAE, TCN, GAN, Diffusion (4种)

## API接口（规划中）

### 1. Trial管理
- `GET /api/trials` - 获取试次列表

### 2. EEG数据
- `GET /api/trials/{trialId}/eeg-data` - 获取EEG数据

### 3. 数据增强
- `POST /api/augmentation/generate` - 生成增强数据
- `POST /api/augmentation/batch-generate` - 批量生成并下载

### 4. 分类预测
- `POST /api/classification/predict` - 分类预测

## 开发进度

- [x] 项目框架搭建
- [x] CORS配置
- [ ] Trial管理接口
- [ ] EEG数据加载
- [ ] 数据预处理
- [ ] 数据增强模型
- [ ] 分类预测模型
- [ ] 批量下载功能

## 技术栈

- **FastAPI**: Web框架
- **Uvicorn**: ASGI服务器
- **Pydantic**: 数据验证
- **NumPy**: 数值计算
- **SciPy**: 信号处理

## 许可证

MIT License
