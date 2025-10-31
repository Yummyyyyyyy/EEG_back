"""
FastAPI 主应用入口
EEG数据处理与分析后端系统
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import trials, augmentation, classification

# 创建 FastAPI 应用实例
app = FastAPI(
    title="EEG Data Processing API",
    description="EEG数据处理、增强与分类预测后端API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # 允许的源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# 注册路由
app.include_router(trials.router)
app.include_router(augmentation.router)
app.include_router(classification.router)


# 根路径 - 健康检查
@app.get("/")
async def root():
    """
    API根路径 - 健康检查
    """
    return {
        "message": "EEG Data Processing API",
        "status": "running",
        "version": "1.0.0"
    }


# API健康检查
@app.get("/api/health")
async def health_check():
    """
    健康检查接口
    """
    return {
        "success": True,
        "message": "API is healthy",
        "data": {
            "status": "ok",
            "version": "1.0.0"
        }
    }


# 启动事件
@app.on_event("startup")
async def startup_event():
    """
    应用启动时执行
    """
    print("=" * 60)
    print("EEG Data Processing API 启动成功!")
    print(f"API文档: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"允许的CORS源: {settings.CORS_ORIGINS}")
    print("=" * 60)


# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭时执行
    """
    print("EEG Data Processing API 关闭")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
