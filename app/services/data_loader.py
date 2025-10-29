"""
EEG数据加载服务
负责从npz文件加载和切分EEG数据
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from app.config import settings
from app.models.trial import Trial
from app.models.eeg import EEGData, EEGChannels

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    EEG数据加载器
    负责加载npz文件，切分trials，提取通道数据
    """

    # Event类型到标签的映射
    EVENT_TO_LABEL = {
        '769': 0,  # left hand
        '770': 1,  # right hand
        '771': 2,  # foot
        '772': 3   # tongue
    }

    # 标签到运动类型的映射
    LABEL_TO_MOTION = {
        0: 'left',
        1: 'right',
        2: 'foot',
        3: 'tongue'
    }

    # MI事件类型列表
    MI_EVENT_TYPES = ['769', '770', '771', '772']

    def __init__(self):
        """初始化数据加载器"""
        self.base_dir = Path(__file__).parent.parent.parent
        self.processed_dir = self.base_dir / "app/data/source/processed"
        self.cleaned_dir = self.base_dir / "app/data/source/cleaned"

        # 内存缓存：存储已加载的npz文件数据
        self._cache: Dict[str, Dict] = {}

        # 预生成trials元数据
        self.trials_metadata = self._generate_trials_metadata()

        logger.info(f"DataLoader initialized")
        logger.info(f"Processed dir: {self.processed_dir}")
        logger.info(f"Cleaned dir: {self.cleaned_dir}")
        logger.info(f"Total trials metadata: {len(self.trials_metadata)}")

    def _generate_trials_metadata(self) -> List[Trial]:
        """
        预生成所有trials的元数据列表
        扫描所有受试者的数据文件，提取trial信息
        """
        trials = []

        # 遍历9个受试者
        for subject_num in range(1, 10):
            subject_id = f"A{subject_num:02d}"

            # 尝试加载该受试者的数据以获取真实的trial信息
            try:
                subject_data = self._load_subject_file(subject_id, "processed")

                # 提取MI events
                mi_events = self._extract_mi_events(subject_data)

                # 为每个MI event创建trial元数据
                for idx, (event_idx, event_type, pos, dur) in enumerate(mi_events):
                    label = self.EVENT_TO_LABEL[event_type]
                    motion_type = self.LABEL_TO_MOTION[label]

                    trial = Trial(
                        id=f"{subject_id}T{idx+1:03d}",
                        subject=subject_num,
                        trialIndex=idx + 1,
                        motionType=motion_type,
                        label=label
                    )
                    trials.append(trial)

                logger.info(f"Loaded {len(mi_events)} trials for subject {subject_id}")

            except Exception as e:
                logger.warning(f"Failed to load metadata for {subject_id}: {e}")
                continue

        return trials

    def _load_subject_file(self, subject_id: str, data_type: str) -> Dict:
        """
        加载受试者的npz文件

        Args:
            subject_id: 受试者ID，如 "A01"
            data_type: 数据类型，"processed" 或 "cleaned"

        Returns:
            包含npz文件所有数据的字典
        """
        cache_key = f"{subject_id}_{data_type}"

        # 检查缓存
        if cache_key in self._cache:
            logger.debug(f"Cache hit: {cache_key}")
            return self._cache[cache_key]

        # 选择文件路径
        if data_type == "processed":
            file_path = self.processed_dir / f"{subject_id}T_continuous_epochs.npz"
        elif data_type == "cleaned":
            file_path = self.cleaned_dir / f"{subject_id}T_continuous_epochs_cleaned.npz"
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading {file_path}")

        # 加载npz文件
        npz_data = np.load(file_path)

        # 转换为普通字典（避免npz文件保持打开）
        data = {
            'eeg_data': npz_data['eeg_data'],
            'eog_data': npz_data['eog_data'],
            'etyp': npz_data['etyp'],
            'epos': npz_data['epos'],
            'edur': npz_data['edur'],
            'artifacts': npz_data['artifacts'],
            'sfreq': float(npz_data['sfreq'])
        }

        npz_data.close()

        # 缓存数据（如果不太大）
        data_size_mb = data['eeg_data'].nbytes / (1024 * 1024)
        if data_size_mb < 100:  # 小于100MB才缓存
            self._cache[cache_key] = data
            logger.info(f"Cached {cache_key} ({data_size_mb:.2f} MB)")

        return data

    def _extract_mi_events(self, subject_data: Dict) -> List[Tuple[int, str, int, int]]:
        """
        从subject数据中提取所有MI事件

        Returns:
            List of (event_index, event_type, position, duration)
        """
        etyp = subject_data['etyp']
        epos = subject_data['epos']
        edur = subject_data['edur']

        mi_events = []
        for i in range(len(etyp)):
            if etyp[i] in self.MI_EVENT_TYPES:
                mi_events.append((i, etyp[i], int(epos[i]), int(edur[i])))

        return mi_events

    def get_all_trials(
        self,
        subject: Optional[int] = None,
        motion_type: Optional[str] = None
    ) -> List[Trial]:
        """
        获取所有trials的列表（支持筛选）

        Args:
            subject: 受试者编号 (1-9)，None表示全部
            motion_type: 运动类型，None表示全部

        Returns:
            Trial列表
        """
        trials = self.trials_metadata.copy()

        # 应用筛选
        if subject is not None:
            trials = [t for t in trials if t.subject == subject]

        if motion_type is not None:
            trials = [t for t in trials if t.motionType == motion_type]

        return trials

    def load_trial_data(
        self,
        trial_id: str,
        extract_mi: bool = False
    ) -> EEGData:
        """
        加载单个trial的EEG数据

        Args:
            trial_id: Trial ID，格式如 "A01T001"
            extract_mi: 是否使用MI提取后的数据（cleaned版本）

        Returns:
            EEGData对象
        """
        # 解析trial_id
        subject_id = trial_id[:3]  # "A01"
        trial_idx = int(trial_id[4:])  # 1
        subject_num = int(subject_id[1:])  # 1

        logger.info(f"Loading trial {trial_id}, extract_mi={extract_mi}")

        # 选择数据类型
        data_type = "cleaned" if extract_mi else "processed"

        # 加载受试者数据
        subject_data = self._load_subject_file(subject_id, data_type)

        # 提取MI events
        mi_events = self._extract_mi_events(subject_data)

        # 检查trial索引是否有效
        if trial_idx < 1 or trial_idx > len(mi_events):
            raise ValueError(
                f"Invalid trial index {trial_idx} for subject {subject_id}. "
                f"Valid range: 1-{len(mi_events)}"
            )

        # 获取该trial的event信息（trial_idx从1开始，数组索引从0开始）
        event_idx, event_type, pos, dur = mi_events[trial_idx - 1]

        # 提取该trial的EEG数据（全22通道）
        eeg_data = subject_data['eeg_data']
        trial_data_full = eeg_data[:, pos:pos+dur]

        # 提取目标5个通道
        channel_data = self._extract_target_channels(trial_data_full)

        # 生成时间标签
        time_labels = self._generate_time_labels(dur, subject_data['sfreq'])

        # 构造Trial信息
        label = self.EVENT_TO_LABEL[event_type]
        motion_type = self.LABEL_TO_MOTION[label]

        trial_info = Trial(
            id=trial_id,
            subject=subject_num,
            trialIndex=trial_idx,
            motionType=motion_type,
            label=label
        )

        # 构造EEGChannels对象
        channels = EEGChannels(
            Fz=channel_data['Fz'].tolist(),
            C3=channel_data['C3'].tolist(),
            Cz=channel_data['Cz'].tolist(),
            C4=channel_data['C4'].tolist(),
            Pz=channel_data['Pz'].tolist()
        )

        # 构造EEGData对象
        eeg_data_obj = EEGData(
            channels=channels,
            labels=time_labels,
            samplingRate=int(subject_data['sfreq']),
            trialInfo=trial_info
        )

        logger.info(f"Successfully loaded trial {trial_id}")

        return eeg_data_obj

    def _extract_target_channels(self, full_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        从22通道数据中提取目标5个通道

        Args:
            full_data: shape (22, n_samples)

        Returns:
            字典，包含5个通道的数据
        """
        channel_indices = settings.CHANNEL_INDICES

        extracted = {}
        for channel_name, idx in channel_indices.items():
            extracted[channel_name] = full_data[idx, :]

        return extracted

    def _generate_time_labels(self, n_samples: int, sampling_rate: float) -> List[float]:
        """
        生成时间标签（秒）

        Args:
            n_samples: 采样点数
            sampling_rate: 采样率

        Returns:
            时间标签列表
        """
        time_labels = [i / sampling_rate for i in range(n_samples)]
        return time_labels

    def clear_cache(self):
        """清空内存缓存"""
        self._cache.clear()
        logger.info("Cache cleared")


# 创建全局单例
data_loader = DataLoader()