from dataclasses import dataclass
from typing import Literal
from pathlib import Path
from einops import rearrange

from torch.utils.data import IterableDataset
from PIL import Image
from torchvision.transforms import ToTensor, Resize



@dataclass
class SingleImageDatasetCfg:
    name: Literal["single_image"]
    data_path: Path
    size: list[int]
    
class SingleImageDataset(IterableDataset):
    cfg: SingleImageDatasetCfg

    def __init__(
        self,
        cfg: SingleImageDatasetCfg,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        
        if cfg.data_path.is_dir():
            img_path = cfg.data_path.glob('*.png') + cfg.data_path.glob('*.jpg') 
            assert len(img_path) == 1, f"Dataset only support single image input but multiple images found as: {img_path}"
            img_path = img_path[0]
        else:
            img_path = cfg.data_path
        
        img = Image.open(img_path).convert('L')
        img = Resize(self.cfg.size)(img)
        self.img = ToTensor()(img)

    def __iter__(self):

        yield {"shadow_map": rearrange(self.img, "() h w -> h w")}
        
    def __len__(self):
        return 1
