from torch.utils.data import Dataset
from .single_image_dataset import SingleImageDataset, SingleImageDatasetCfg

DATASETS: dict[str, Dataset] = {
    "single_image": SingleImageDataset,
}

DatasetCfg = SingleImageDatasetCfg

def get_dataset(
    cfg: DatasetCfg,
) -> Dataset:

    return DATASETS[cfg.name](cfg)