from omegaconf import DictConfig, OmegaConf
import hydra
import hydra

import os

import torch
import numpy as np

from data.neurobench.dataloader import get_dataloader
from efficient_rsnn_bmi.utils.logger import get_logger
from data.config.dataloader import DatasetLoaderConfig
from efficient_rsnn_bmi.utils.helper import from_config

logger = get_logger("train-tinyRSNN")

@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def train_rsnn_tiny(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    logger.info("Starting new simulation...")

    dtype = getattr(torch, cfg.dtype)

    # to make the experiment consistent, the initial weight will be make fixed
    if cfg.seed:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # if using GPU, make sure initialize the weight that is used by GPU
        if torch.cuda.is_available and cfg.node:
            torch.cuda.manual_seed(cfg.seed)
            # Ensure exact repeatability
            torch.backends.cudnn.deterministic = True # Ensures always using the same computation method
            torch.backends.cudnn.benchmark = False # Disables cuDNN’s automatic selection of the fastest computation method

    logger.info(f"Config: {cfg}")
    # Get DataLoader
    dataloader = get_dataloader(from_config(cfg.datasets, DatasetLoaderConfig), dtype=dtype)

    for monkey_name in cfg.train_monkeys:
        nb_inputs = cfg.datasets.nb_inputs[monkey_name]
        logger.info(f"Training on monkey: {monkey_name}")

        if cfg.pretraining:
            filename = list(cfg.datasets.pretrain_filenames[monkey_name].values())

            logger.info("Constructing model for " + monkey_name + " pretraining...")
            pretrain_dat, pretrain_val_dat, _ = dataloader.get_multiple_sessions_data(
                filename
            )

            print("Training", pretrain_dat)
            print("Validation", pretrain_val_dat)
            
if __name__ == "__main__":
    train_rsnn_tiny()