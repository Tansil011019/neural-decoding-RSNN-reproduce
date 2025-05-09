import logging

from experiments.models.rsnn.rsnn import BaseRecurrentSpikingModel

logger = logging.getLogger(__name__)

def get_model (cfg, nb_inputs, dtype, data=None):
    """
    Get the model based on the configuration.
    """
    nb_time_steps = int(cfg.datasets.sample_duration / cfg.datasets.stride)
    nb_outputs = cfg.datasets.nb_outputs
    
    model = BaseRecurrentSpikingModel(
        
    )