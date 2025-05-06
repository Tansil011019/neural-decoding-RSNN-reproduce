import torch
from torch.utils.data import Dataset
import numpy as np
import stork

from neurobench.datasets.primate_reaching import PrimateReaching
from neurobench.datasets.utils import download_url
from urllib.error import URLError

from data.config.dataloader import DatasetLoaderConfig, PretrainPrimateReachingConfig

import math
import os

from efficient_rsnn_bmi.utils.logger import get_logger

logger = get_logger(__name__)

def get_dataloader(cfg: DatasetLoaderConfig, dtype: torch.dtype = torch.float32) -> Dataset:
    logger.info("Loading dataset...")
    logger.info(f"Dataset config: {cfg}")
    dataloader = DatasetLoader(config=cfg)
    logger.info("Dataset loaded")
    return dataloader

class DatasetLoader(Dataset):
    """
    Loads the data from the PrimateReaching Dataset
    """
    def __init__(
        self,
        config: DatasetLoaderConfig
    ):
        self.config = config
        assert self.config.file_path is not None, "File path must be provided"
        assert self.config.val_ratio > ((1 - self.config.train_ratio) / 2), "Validation ratio must be less than half of the remaining data after training ratio is applied"
        self.n_time_steps = int(self.config.sample_duration / self.config.stride)
    
    def get_multiple_sessions_data(self, filenames: list[str]):

        ds_train, ds_val, ds_test = [], [], []
        for filename in filenames:
            monkey_train, monkey_val, monkey_test = self.get_single_session_data(filename)
            ds_train.append(monkey_train)
            ds_val.append(monkey_val)
            ds_test.append(monkey_test)
        dataset_train = torch.utils.data.ConcatDataset(ds_train)
        dataset_val = torch.utils.data.ConcatDataset(ds_val)

        return dataset_train, dataset_val, ds_test

    def get_single_session_data(self, filename: str):
        training_config: PretrainPrimateReachingConfig = {
            "file_path": self.config.file_path,
            "filename": filename,
            "num_steps": self.config.num_steps,
            "train_ratio": self.config.train_ratio,
            "label_series": self.config.label_series,
            "biological_delay": self.config.biological_delay,
            "spike_sorting": self.config.spike_sorting,
            "stride": self.config.stride,
            "bin_width": self.config.bin_width,
            "max_segment_length": self.config.max_segment_length,
            "split_num": self.config.split_num,
            "remove_segments_inactive": self.config.remove_segments_inactive,
        }

        test_config: PretrainPrimateReachingConfig= {
            "file_path": self.config.file_path,
            "filename": filename,
            "num_steps": self.config.num_steps,
            "train_ratio": self.config.train_ratio,
            "label_series": self.config.label_series,
            "biological_delay": self.config.biological_delay,
            "spike_sorting": self.config.spike_sorting,
            "stride": self.config.stride,
            "bin_width": self.config.bin_width,
            "max_segment_length": self.config.max_segment_length,
            "split_num": self.config.split_num,
            "remove_segments_inactive": False
        }

        dataset = PretrainPrimateReachingDataset(
            config=training_config
        )

        # * This is for generalize testing code with inactive segments
        if self.config.remove_segments_inactive:
            dataset_test = PretrainPrimateReachingDataset(
                config=test_config
            )
        else:
            dataset_test = dataset

        # * This is for customization validation size (Resplit)
        ind_train_val = dataset.ind_train + dataset.ind_val

        eff_ratio_val = self.config.val_ratio / (math.ceil(len(ind_train_val) / len(dataset)*100) / 100)

        n_val = int(np.round(len(dataset) * eff_ratio_val))
        
        # TODO: I think this can be more randomized
        if self.config.random_val:
            start_idx = np.random.choice(a=ind_train_val[:-n_val], size=1)[0]
            ind_val = np.array(ind_train_val[start_idx:start_idx + n_val])
            ind_train = np.array(sorted(set(ind_train_val) - set(ind_val)))
        else:
            ind_train = np.array(ind_train_val[:-n_val])
            ind_val = np.array(sorted(set(ind_train_val) - set(ind_train)))

        spikes = dataset.samples.T
        labels = dataset.labels.T

        spikes_test = dataset_test.samples.T
        labels_test = dataset_test.labels.T

        self.ind_train = ind_train
        self.ind_val = ind_val
        self.ind_test = dataset_test.ind_test

        spikes_train = spikes[ind_train]
        spikes_val = spikes[ind_val]
        spikes_test = spikes_test[self.ind_test]

        labels_train = labels[ind_train]
        labels_val = labels[ind_val]
        labels_test = labels_test[self.ind_test]

        if self.config.extend_data:
            logger.info(f"Extending data with {self.n_time_steps} time steps")
            train_data, train_labels = self.extend_spikes(spikes_train, labels_train, self.n_time_steps, self.config.split_num)
            val_data, val_labels = self.extend_spikes(spikes_val, labels_val, self.n_time_steps, self.config.split_num)
        else: 
            train_data, train_labels = self.extend_spikes(spikes_train, labels_train, chunks=99)
            val_data, val_labels = self.extend_spikes(spikes_val, labels_val, chunks=99)
        
        test_data = [spikes_test]
        test_labels = [labels_test]

        test_data = torch.stack(test_data)
        test_labels = torch.stack(test_labels)

        if any([self.config.p_drop > 0, self.config.p_insert > 0, self.config.jitter_sigma > 0]):
            data_augmentation_kwargs = dict(
                data_augmentation = True,
                p_drop = self.config.p_drop,
                p_insert = self.config.p_insert,
                jitter_sigma = self.config.jitter_sigma,
            )
        else:
            data_augmentation_kwargs = {}
        
        train_ras_data = self.to_ras(train_data, train_labels, **data_augmentation_kwargs)
        val_ras_data = self.to_ras(val_data, val_labels)
        test_ras_data = self.to_ras(test_data, test_labels)

        return train_ras_data, val_ras_data, test_ras_data

    def extend_spikes(self, spikes, labels, chunks = "all", chunk_size=100):
        """
        Extend the spikes and labels data by chunking them into smaller segments.
        """
        if chunks == 'all':
            chunks = self.n_time_steps

        extended_spikes = []
        extended_labels = []

        for i in range(0, chunks, chunk_size):
            curr_spikes = spikes[i:]
            curr_labels = labels[i:]

            splitter = np.arange(self.n_time_steps, curr_spikes.shape[0], self.n_time_steps)

            extended_spikes += np.split(curr_spikes, splitter)[:-1]
            extended_labels += np.split(curr_labels, splitter)[:-1]

        extended_spikes = torch.stack(extended_spikes)
        extended_labels = torch.stack(extended_labels)

        return extended_spikes, extended_labels
    
    def to_ras(self, data, labels, **data_augmentation_kwargs):
        """
        Convert the data to RAS format.
        """
        ras_data = [[[], []] for _ in data]

        for i, sample in enumerate(data):
            for j in range(sample.shape[-1]):
                spike_times = np.where(sample[:, j] == 1)[0].tolist()
                ras_data[i][0] += spike_times
                ras_data[i][1] += [j] * len(spike_times)
            ras_data[i] = torch.tensor(ras_data[i], dtype=self.config.dtype)
        
        monkey_ds_kwargs = dict(nb_steps = data.shape[-2], nb_units=data.shape[-1], time_scale=1.0)

        monkey_ds = stork.datasets.RasDataset(
            (ras_data, labels), dtype=self.config.dtype, **monkey_ds_kwargs, **data_augmentation_kwargs
        )

        return monkey_ds

class PretrainPrimateReachingDataset(PrimateReaching):
    """
    Loads the data from the PrimateReaching Dataset
    """
    def __init__(
        self,
        config: PretrainPrimateReachingConfig
    ):
        super().__init__(
            file_path=config['file_path'],
            filename=config['filename'],
            num_steps=config['num_steps'],
            train_ratio=config['train_ratio'],
            label_series=config['label_series'],
            biological_delay=config['biological_delay'],
            spike_sorting=config['spike_sorting'],
            stride=config['stride'],
            bin_width=config['bin_width'],
            max_segment_length=config['max_segment_length'],
            split_num=config['split_num'],
            remove_segments_inactive=config['remove_segments_inactive'],
        )
        
    def download(self):
        """
        Download the dataset if it is not already present.
        """
        if self.filename in self.md5s.keys():
            md5 = self.md5s[self.filename]
        else:
            md5 = None

        if self._check_exists(self.file_path, md5):
            return

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # download file
        url = f"{self.url}{self.filename}"
        try:
            print(f"Downloading {url}")
            download_url(url, self.file_path, md5=md5)
        except URLError as error:
            print(f"Failed to download (trying next):\n{error}")
        finally:
            print()