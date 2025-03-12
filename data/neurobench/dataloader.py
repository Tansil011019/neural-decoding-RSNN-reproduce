import torch
from torch.utils.data import Dataset

from neurobench.datasets.primate_reaching import PrimateReaching

def get_dataloader(cfg, dtype=torch.float32):
    dataloader = DatasetLoader()

class DatasetLoader(Dataset):
    """
    Loads the data from the PrimateReaching Dataset
    """
    def __init__(
        self,
        file_path: str,
        filename: str,
        num_steps: int = 1,
        train_ratio: float = 0.8,
        label_series: bool = False,
        biological_delay: int = 0,
        spike_sorting: bool = False,
        stride: float = 0.004,
        bin_width: float = 0.028, # dif here is the time step use is the same with stride
        max_segment_length: int = 2000,
        split_num: int = 1,
        remove_segments_inactive: bool = False,
        download: bool = False,
        random_val: bool = False,
        extend_data: bool = True,
        sample_duration: int = 2,
        dtype = torch.float32,
        p_drop: float = 0.0,
        p_insert: float = 0.0,
        jitter_sigma: float = 0.0
    ):
        self.file_path = file_path
        self.filename = filename
        self.num_steps = num_steps
        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = 1 - train_ratio
        self.biological_delay = biological_delay
        self.spike_sorting = spike_sorting
        self.label_series = label_series
        self.random_val = random_val
        self.extend_data = extend_data
        self.sample_duration = sample_duration
        self.remove_segments_inactive = remove_segments_inactive
        self.dtype = dtype
        self.p_drop = p_drop
        self.p_insert = p_insert
        self.jitter_sigma = jitter_sigma
        self.stride = stride
        self.bin_width = bin_width
        self.download = download
        self.max_segment_length = max_segment_length
        self.split_num = split_num

class PretrainPrimateReaching(PrimateReaching):
    def __init__(
        self,
        file_path: str,
        filename: str,
        num_steps: int = 1,
        train_ratio: float = 0.8,
        label_series: bool = False,
        biological_delay: int = 0,
        spike_sorting: bool = False,
        stride: float = 0.004,
        bin_width: float = 0.028,
        max_segment_length: int = 2000,
        split_num: int = 1,
        remove_segments_inactive: bool = False,
        download: bool = False
    ):
        pass