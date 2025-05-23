import numpy as np
import torch

from neurobench.models import TorchModel as StorkModel
from neurobench.benchmarks import Benchmark as StorkBenchmark
from neurobench.metrics.static import Footprint, ConnectionSparsity
from neurobench.metrics.workload import R2 as RSquared, ActivationSparsity, SynapticOperations

from .loss import _choose_loss

def get_test_loss(cfg):

    loss_class = _choose_loss(
        cfg,
    )
    return loss_class()

def _configure_model_eval(model, test_dat, cfg):

    model.set_nb_steps(test_dat[0][0].shape[0])
    model.loss_stack = get_test_loss(cfg)

    return model

def evaluate_model(model, cfg, test_dat):

    # Re-configure model for eval
    model = _configure_model_eval(model, test_dat, cfg)

    # Evaluate model
    scores, pred, _ = model.evaluate_continuous_test_data(test_dat)

    target = test_dat[0][1].numpy()

    SST = np.sum((target - np.mean(target, 0)) ** 2, 0)
    SSR = np.sum((target - pred) ** 2, 0)
    R2 = 1 - SSR / SST

    bm = {}
    bm["R2 X (JR)"] = R2[0].astype(float)
    bm["R2 Y (JR)"] = R2[1].astype(float)
    bm["R2 mean (JR)"] = np.mean(R2).astype(float)

    # BENCHMARK MODEL
    # # # # # # # # #
    # convert to stork model
    test_model = StorkModel(model)

    # define metrics to benchmark

    # Benchmark expects the following dataloader
    test_set_loader = torch.utils.data.DataLoader(
        test_dat,
        batch_size=1,
        shuffle=False,
    )

    metric_registry = {
        "footprint": Footprint,
        "connection_sparsity": ConnectionSparsity,
        "activation_sparsity": ActivationSparsity,
        "synaptic_operations": SynapticOperations,
        "r2": RSquared,
    }

    static_metrics = [metric_registry[m] for m in cfg.evaluation.static_metrics]
    workload_metrics = [metric_registry[m] for m in cfg.evaluation.workload_metrics]

    benchmark = StorkBenchmark(
        test_model,
        test_set_loader,
        [],
        [],
        [static_metrics, workload_metrics],
    )
    bm_results = benchmark.run()
    bm.update(bm_results)

    return model, scores, pred, bm

def benchmark_model(model, cfg, test_dat):

    # Re-configure model for eval
    model = _configure_model_eval(model, test_dat, cfg)
    
    test_model = StorkModel(model)

    # Benchmark expects the following dataloader
    test_set_loader = torch.utils.data.DataLoader(
        test_dat,
        batch_size=1,
        shuffle=False,
    )

    metric_registry = {
        "footprint": Footprint,
        "connection_sparsity": ConnectionSparsity,
        "activation_sparsity": ActivationSparsity,
        "synaptic_operations": SynapticOperations,
        "r2": RSquared,
    }

    static_metrics = [metric_registry[m] for m in cfg.evaluation.static_metrics]
    workload_metrics = [metric_registry[m] for m in cfg.evaluation.workload_metrics]

    benchmark = StorkBenchmark(
        test_model,
        test_set_loader,
        [],
        [],
        [static_metrics, workload_metrics],
    )
    bm_results = benchmark.run()
    
    return bm_results