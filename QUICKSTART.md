# 快速开始指南

## 安装与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务器

**方法1：使用启动脚本（推荐）**
```bash
./start.sh
```

**方法2：使用 uvicorn 命令**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

**方法3：直接运行 Python 模块**
```bash
python -m app.main
```

### 3. 访问API

服务器启动后，访问以下地址：

- **Swagger API文档**: http://localhost:8001/docs
- **ReDoc文档**: http://localhost:8001/redoc
- **根路径**: http://localhost:8001/
- **健康检查**: http://localhost:8001/api/health

## 测试API

使用 curl 测试：

```bash
# 健康检查
curl http://localhost:8001/api/health

# 根路径
curl http://localhost:8001/
```

## CORS配置

后端已配置CORS，允许以下前端源访问：
- http://localhost:3001 (前端主端口)
- http://127.0.0.1:3001
- http://localhost:3000
- http://127.0.0.1:3000

## 项目结构

```
EEG_back/
├── app/
│   ├── main.py           # 应用入口
│   ├── config.py         # 配置文件
│   ├── api/              # API路由（待开发）
│   ├── models/           # 数据模型（待开发）
│   ├── services/         # 业务逻辑（待开发）
│   └── ml_models/        # ML模型（待开发）
├── requirements.txt      # 依赖包
├── start.sh             # 启动脚本
└── README.md            # 项目说明
```

## 下一步

- [ ] 实现 Trial 管理接口
- [ ] 实现 EEG 数据加载
- [ ] 实现数据预处理功能
- [ ] 实现数据增强模型
- [ ] 实现分类预测功能

## 常见问题

**Q: 端口8001被占用怎么办？**
A: 修改 `app/config.py` 中的 `PORT` 值，或在启动命令中指定其他端口：
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

**Q: 如何关闭服务器？**
A: 在终端按 `Ctrl+C`

**Q: 如何查看日志？**
A: 日志会直接输出到终端，或者查看 `logs/` 目录（如果配置了）