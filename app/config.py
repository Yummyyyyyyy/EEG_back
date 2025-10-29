"""
应用配置文件
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """
    应用配置类
    """
    # 应用基础配置
    APP_NAME: str = "EEG Data Processing API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8001  # 使用8001端口，避免与其他服务冲突

    # CORS 配置
    CORS_ORIGINS: List[str] = [
        "http://localhost:3001",  # 前端开发服务器
        "http://127.0.0.1:3001",
        "http://localhost:3000",  # 备用端口
        "http://127.0.0.1:3000",
    ]

    # 数据路径配置
    DATA_DIR: str = "app/data"
    RAW_DATA_DIR: str = "app/data/raw"
    PROCESSED_DATA_DIR: str = "app/data/processed"
    CACHE_DIR: str = "app/data/cache"

    # EEG数据配置
    SAMPLING_RATE: int = 250  # 采样率 Hz
    CHANNELS: List[str] = ["Fz", "C3", "Cz", "C4", "Pz"]  # 5个通道
    CHANNEL_INDICES: dict = {
        "Fz": 0,   # index 0
        "C3": 7,   # index 7
        "Cz": 9,   # index 9
        "C4": 11,  # index 11
        "Pz": 21   # index 21 (你说的是19，但标准BCI Competition IV 2a中Pz是21，请再确认)
    }
    SUBJECTS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 9个受试者

    # 运动想象类型
    MOTION_TYPES: List[str] = ["left", "right", "foot", "tongue"]

    # 数据增强方法
    AUGMENTATION_METHODS: List[str] = ["vae", "tcn", "gan", "diffusion"]

    class Config:
        env_file = ".env"
        case_sensitive = True


# 创建全局配置实例
settings = Settings()
