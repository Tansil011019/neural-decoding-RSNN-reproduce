import torch

def get_lr_scheduler(cfg, opt):
    
    if cfg.training.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        scheduler_kwargs = {"T_max": cfg.training.nb_epochs_train}
    else:
        raise ValueError(f"Unknown lr_scheduler: {cfg.training.lr_scheduler}")
    
    return scheduler, scheduler_kwargs