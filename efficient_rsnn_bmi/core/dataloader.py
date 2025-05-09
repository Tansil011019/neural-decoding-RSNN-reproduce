import torch
from torch.utils.data import Dataset

from data.config.dataloader import DatasetLoaderConfig
from data.neurobench.dataloader import DatasetLoader

from efficient_rsnn_bmi.utils.logger import get_logger

logger = get_logger(__name__)

def get_dataloader(cfg: DatasetLoaderConfig, dtype: torch.dtype = torch.float32) -> Dataset:
    logger.info("Loading dataset...")
    logger.info(f"Dataset config: {cfg}")
    dataloader = DatasetLoader(config=cfg)
    logger.info("Dataset loaded")
    return dataloader

def compute_input_firing_rates(data, cfg):
    mean1 = 0
    mean2 = 0

    for i in range(len(data)):
        mean1 += torch.sum(data[i][0][:, :96]) / cfg.datasets.sample_duration / 96
        try:
            mean2 += torch.sum(data[i][0][:, 96:]) / cfg.datasets.sample_duration / 96
        except:
            continue

    mean1 /= len(data)
    mean2 /= len(data)

    # For LOCO
    if data[0][0].shape[1] == 192:
        return mean1, mean2

    # FOR INDY
    else:
        return mean1, None