import stork

def get_regularizers(cfg):
    regs = []
    regLB = stork.regularizers.LowerBoundL2(
        cfg.training.LB_L2_strength, threshold=cfg.training.LB_L2_thresh, dims=False
    )
    regs.append(regLB)
    regUB = stork.regularizers.UpperBoundL2(
        cfg.training.UB_L2_strength, threshold=cfg.training.UB_L2_thresh, dims=1
    )
    regs.append(regUB)
    return regs